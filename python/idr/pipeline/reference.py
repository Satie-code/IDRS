"""Reference (target) construction for the future forward-velocity model.

Phase 2 does not train anything. It decides, and records, *where a target value
would come from* and how much it can be trusted — so that a later phase cannot
silently pick up a low-quality label.

The hierarchy, highest first:

``TIER_1_VEHICLE_SYNCED``
    Vehicle velocity from a pair whose alignment was confirmed by signal
    correlation. The vehicle GNSS genuinely updates at ~10 Hz.

``TIER_2_VEHICLE_CLOCK``
    Vehicle velocity where only the clocks agree and no signal confirmation was
    possible. Usable, but the alignment is an inference.

``TIER_3_SMARTPHONE_GNSS``
    Smartphone GNSS speed. Marked low-confidence by construction: Phase 1
    showed the smartphone position is forward-filled at ~0.1 Hz, so this is a
    sparse signal wearing a 10 Hz costume. Only offered where nothing better
    exists, and only on rows whose position is fresh.

``UNAVAILABLE``
    No defensible target. Reported, never guessed.

Every emitted label carries its source, quality, and availability, so label
provenance survives into training.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from idr.pipeline.canonical import QualityFlag


class LabelSource(StrEnum):
    TIER_1_VEHICLE_SYNCED = "TIER_1_VEHICLE_SYNCED"
    TIER_2_VEHICLE_CLOCK = "TIER_2_VEHICLE_CLOCK"
    TIER_3_SMARTPHONE_GNSS = "TIER_3_SMARTPHONE_GNSS"
    UNAVAILABLE = "UNAVAILABLE"


#: Confidence attached to each tier. Kept as data so the report can print it.
TIER_QUALITY: dict[str, str] = {
    LabelSource.TIER_1_VEHICLE_SYNCED: "high",
    LabelSource.TIER_2_VEHICLE_CLOCK: "medium",
    LabelSource.TIER_3_SMARTPHONE_GNSS: "low",
    LabelSource.UNAVAILABLE: "none",
}

#: Beyond this separation from the nearest reference sample, a target is not
#: emitted: interpolating further would be inventing a value.
MAX_INTERPOLATION_GAP_S = 0.5


@dataclass
class LabelCoverage:
    """How well one segment could be labelled."""

    segment_id: str
    session_id: str
    source_file: str
    label_source: str
    label_quality: str
    row_count: int
    labelled_rows: int
    coverage_fraction: float
    mean_label_mps: float | None
    max_label_mps: float | None
    stationary_fraction: float | None
    note: str = ""


def build_velocity_labels(
    phone_frame: pd.DataFrame,
    *,
    segment_id: str,
    session_id: str,
    source_file: str,
    vehicle_time_s: np.ndarray | None = None,
    vehicle_speed_mps: np.ndarray | None = None,
    sync_offset_s: float | None = None,
    sync_confirmed: bool = False,
) -> tuple[pd.DataFrame, LabelCoverage]:
    """Attach ``ref_velocity_mps`` (+ provenance columns) to a smartphone frame.

    ``vehicle_time_s`` is on the vehicle's time-of-day clock; the smartphone's
    time-of-day axis is shifted by ``sync_offset_s`` to meet it. Values are
    linearly interpolated onto smartphone sample times, but **only where a
    reference sample is within** :data:`MAX_INTERPOLATION_GAP_S`; anything
    further is left unlabelled rather than extrapolated.
    """
    frame = phone_frame.copy()
    row_count = len(frame)
    labels = np.full(row_count, np.nan)
    note = ""

    have_vehicle = (
        vehicle_time_s is not None
        and vehicle_speed_mps is not None
        and sync_offset_s is not None
        and bool(np.isfinite(sync_offset_s))
    )

    if have_vehicle:
        assert vehicle_time_s is not None and vehicle_speed_mps is not None
        assert sync_offset_s is not None
        source = (
            LabelSource.TIER_1_VEHICLE_SYNCED
            if sync_confirmed
            else LabelSource.TIER_2_VEHICLE_CLOCK
        )
        phone_clock = frame["time_of_day_s"].to_numpy(dtype="float64") - float(sync_offset_s)
        ref_finite = np.isfinite(vehicle_time_s) & np.isfinite(vehicle_speed_mps)
        if ref_finite.sum() >= 2:
            ref_t = vehicle_time_s[ref_finite]
            ref_v = vehicle_speed_mps[ref_finite]
            order = np.argsort(ref_t)
            ref_t, ref_v = ref_t[order], ref_v[order]

            interpolated = np.interp(phone_clock, ref_t, ref_v, left=np.nan, right=np.nan)
            # Reject points whose nearest reference sample is too far away.
            next_idx = np.clip(np.searchsorted(ref_t, phone_clock), 0, ref_t.size - 1)
            nearest = np.abs(ref_t[next_idx] - phone_clock)
            previous_idx = np.clip(np.searchsorted(ref_t, phone_clock) - 1, 0, ref_t.size - 1)
            nearest = np.minimum(nearest, np.abs(ref_t[previous_idx] - phone_clock))
            usable = np.isfinite(interpolated) & (nearest <= MAX_INTERPOLATION_GAP_S)
            labels = np.where(usable, interpolated, np.nan)
            note = (
                f"interpolated from vehicle velocity at offset {sync_offset_s:.3f}s; "
                f"{int((~usable).sum())} row(s) left unlabelled beyond "
                f"{MAX_INTERPOLATION_GAP_S}s of a reference sample"
            )
        else:
            source = LabelSource.UNAVAILABLE
            note = "paired vehicle stream had fewer than 2 usable velocity samples"
    else:
        # Fall back to the smartphone's own GNSS speed, but only on fresh rows.
        speed = frame["gnss_speed_mps"].to_numpy(dtype="float64")
        fresh = frame["gnss_position_fresh"].fillna(False).to_numpy(dtype=bool)
        if np.isfinite(speed).any() and fresh.any():
            source = LabelSource.TIER_3_SMARTPHONE_GNSS
            labels = np.where(fresh & np.isfinite(speed), speed, np.nan)
            note = (
                "no synchronized vehicle reference; smartphone GNSS speed used on fresh "
                "rows only. Low confidence: the smartphone position updates ~0.1 Hz."
            )
        else:
            source = LabelSource.UNAVAILABLE
            note = "no vehicle reference and no usable smartphone GNSS speed"

    available = np.isfinite(labels)
    frame["ref_velocity_mps"] = labels.astype("float32")
    frame["label_source"] = str(source)
    frame["label_quality"] = TIER_QUALITY[source]
    frame["label_available"] = pd.array(available, dtype="boolean")

    if source in (LabelSource.TIER_3_SMARTPHONE_GNSS, LabelSource.UNAVAILABLE):
        flags = frame["quality_flags"].to_numpy(dtype="int64").copy()
        flags |= QualityFlag.REFERENCE_LOW_CONFIDENCE.value
        frame["quality_flags"] = flags.astype("int32")

    finite_labels = labels[available]
    coverage = LabelCoverage(
        segment_id=segment_id,
        session_id=session_id,
        source_file=source_file,
        label_source=str(source),
        label_quality=TIER_QUALITY[source],
        row_count=row_count,
        labelled_rows=int(available.sum()),
        coverage_fraction=float(available.mean()) if row_count else 0.0,
        mean_label_mps=float(finite_labels.mean()) if finite_labels.size else None,
        max_label_mps=float(finite_labels.max()) if finite_labels.size else None,
        stationary_fraction=(float((finite_labels < 0.5).mean()) if finite_labels.size else None),
        note=note,
    )
    return frame, coverage
