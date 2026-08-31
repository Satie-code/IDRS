"""Magnetometer trust assessment.

A magnetometer in a car is measuring the car as much as the Earth. Door
motors, speakers, the chassis itself and any nearby steel structure all add a
field, and the added field is often *steady* in the phone frame, which makes it
look like a perfectly good heading reference right up until the vehicle turns.

This module never corrects a magnetometer reading and never calibrates one. It
answers a narrower question — how much should the orientation estimator lean on
this measurement? — and returns the individual reasons alongside the answer, so
"the magnetometer was ignored" is always traceable to a specific check.

There is deliberately no learned anomaly model here. That would be a Phase 5+
question, and it would need this infrastructure to label its training data
anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntFlag

import numpy as np

from idr.frames.conventions import (
    EARTH_FIELD_MAX_UT,
    EARTH_FIELD_MIN_UT,
    MIN_VECTOR_NORM,
)
from idr.frames.gravity import GravityEstimate
from idr.frames.vectors import VectorSeries

#: Largest sane rate of change of field direction, rad/s. The Earth's field is
#: fixed; apparent rotation comes from the device turning. 3 rad/s ≈ 170°/s is
#: far faster than a vehicle turns and generously above a handheld sweep, so
#: exceeding it points at interference or a dropped sample rather than motion.
MAX_DIRECTION_RATE_RPS = 3.0

#: Fraction of samples that must pass every check before the magnetometer is
#: trusted without qualification.
GOOD_SAMPLE_FRACTION = 0.8

#: Magnetic dip varies with latitude from 0° at the magnetic equator to ±90° at
#: the poles, so dip alone cannot be checked against a fixed value. What *can*
#: be checked is stability: within one short window the dip should barely move.
#: 15° of variation within a window means the local field is being disturbed.
MAX_DIP_VARIATION_DEG = 15.0


class MagnetometerIssue(IntFlag):
    """Why a magnetometer sample or window is not fully trusted."""

    NONE = 0
    #: Field strength outside the plausible range for Earth's surface field.
    MAGNITUDE_OUT_OF_RANGE = 1
    #: Direction changing faster than any real device motion explains.
    ABRUPT_DIRECTION_CHANGE = 2
    #: Magnitude varying far more than a rigid rotation in a fixed field would.
    UNSTABLE_MAGNITUDE = 4
    #: Inclination relative to gravity wandering — a disturbed local field.
    DIP_INCONSISTENT = 8
    #: Channel absent, all-NaN, or too short to assess.
    UNUSABLE = 16


@dataclass
class MagnetometerQuality:
    """Per-window magnetometer assessment.

    ``weight`` is what the orientation estimator multiplies its magnetic
    heading correction by. It is 0 when the field cannot be trusted at all,
    which is a normal outcome, not an error.
    """

    weight: float
    issues: MagnetometerIssue
    sample_count: int
    usable_count: int
    good_fraction: float
    median_magnitude_ut: float | None
    magnitude_cv: float | None
    max_direction_rate_rps: float | None
    dip_variation_deg: float | None
    notes: list[str] = field(default_factory=list)

    @property
    def is_trusted(self) -> bool:
        return self.weight > 0.0

    def describe(self) -> str:
        if self.issues == MagnetometerIssue.NONE:
            return "no issues detected"
        return ", ".join(
            (issue.name or str(issue)).lower().replace("_", " ")
            for issue in MagnetometerIssue
            if issue is not MagnetometerIssue.NONE and issue & self.issues
        )


def _unusable(sample_count: int, reason: str) -> MagnetometerQuality:
    return MagnetometerQuality(
        weight=0.0,
        issues=MagnetometerIssue.UNUSABLE,
        sample_count=sample_count,
        usable_count=0,
        good_fraction=0.0,
        median_magnitude_ut=None,
        magnitude_cv=None,
        max_direction_rate_rps=None,
        dip_variation_deg=None,
        notes=[reason],
    )


def _direction_rates(samples: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Angular rate of the field direction between consecutive samples, rad/s."""
    norms = np.linalg.norm(samples, axis=1)
    usable = norms > MIN_VECTOR_NORM
    if usable.sum() < 2:
        return np.zeros(0, dtype="float64")
    unit = samples[usable] / norms[usable, None]
    stamps = times[usable]
    dt = np.diff(stamps)
    # A zero or negative dt cannot yield a rate; Phase 2 flags duplicate
    # timestamps rather than dropping them, so they reach here legitimately.
    valid = dt > 0.0
    if not valid.any():
        return np.zeros(0, dtype="float64")
    dots = np.clip(np.einsum("ij,ij->i", unit[:-1], unit[1:]), -1.0, 1.0)
    angles = np.arccos(dots)
    return angles[valid] / dt[valid]


def _dip_angles_deg(samples: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Angle between the field and the local down direction, per sample."""
    norms = np.linalg.norm(samples, axis=1)
    usable = norms > MIN_VECTOR_NORM
    if not usable.any():
        return np.zeros(0, dtype="float64")
    unit = samples[usable] / norms[usable, None]
    cosines = np.clip(unit @ down, -1.0, 1.0)
    return np.degrees(np.arccos(cosines))


def assess_magnetometer(
    mag: VectorSeries | None,
    *,
    gravity: GravityEstimate | None = None,
    available: bool = True,
) -> MagnetometerQuality:
    """Score a magnetometer window and explain the score.

    ``available`` carries Phase 2's structural availability flag. A file that
    never had a magnetometer is a different situation from one whose
    magnetometer was disturbed, and collapsing the two would hide which files
    could be improved by better filtering and which simply lack the sensor.
    """
    if not available:
        return _unusable(0, "no magnetometer channel in this source file")
    if mag is None or len(mag) == 0:
        return _unusable(0 if mag is None else len(mag), "no magnetometer samples")

    finite = mag.finite_mask
    usable_count = int(finite.sum())
    if usable_count < 2:
        return _unusable(len(mag), f"only {usable_count} finite magnetometer sample(s)")

    usable = mag.subset(finite)
    magnitudes = usable.norms
    positive = magnitudes > MIN_VECTOR_NORM
    if not positive.any():
        return _unusable(len(mag), "all magnetometer samples have zero magnitude")

    issues = MagnetometerIssue.NONE
    notes: list[str] = []

    in_range = (magnitudes >= EARTH_FIELD_MIN_UT) & (magnitudes <= EARTH_FIELD_MAX_UT)
    good_fraction = float(in_range.mean())
    median_magnitude = float(np.median(magnitudes))
    if good_fraction < GOOD_SAMPLE_FRACTION:
        issues |= MagnetometerIssue.MAGNITUDE_OUT_OF_RANGE
        notes.append(
            f"{100.0 * (1.0 - good_fraction):.1f}% of samples fall outside "
            f"{EARTH_FIELD_MIN_UT:.0f}–{EARTH_FIELD_MAX_UT:.0f} µT "
            f"(median {median_magnitude:.1f} µT)"
        )

    mean_magnitude = float(np.mean(magnitudes))
    magnitude_cv = (
        float(np.std(magnitudes) / mean_magnitude) if mean_magnitude > MIN_VECTOR_NORM else None
    )
    # A rigid rotation in a fixed field leaves the magnitude unchanged, so a
    # large coefficient of variation is evidence of a field that is not fixed.
    if magnitude_cv is not None and magnitude_cv > 0.15:
        issues |= MagnetometerIssue.UNSTABLE_MAGNITUDE
        notes.append(
            f"field strength varies by {100.0 * magnitude_cv:.1f}% within the window; "
            "a rigid rotation in a fixed field would not change it"
        )

    rates = _direction_rates(usable.samples, usable.analysis_time_s)
    max_rate = float(np.max(rates)) if rates.size else None
    if max_rate is not None and max_rate > MAX_DIRECTION_RATE_RPS:
        issues |= MagnetometerIssue.ABRUPT_DIRECTION_CHANGE
        notes.append(
            f"field direction slews at up to {math.degrees(max_rate):.0f}°/s, "
            "faster than device motion explains"
        )

    dip_variation: float | None = None
    if gravity is not None and gravity.direction is not None:
        dips = _dip_angles_deg(usable.samples, gravity.direction.to_array())
        if dips.size:
            dip_variation = float(np.percentile(dips, 95) - np.percentile(dips, 5))
            if dip_variation > MAX_DIP_VARIATION_DEG:
                issues |= MagnetometerIssue.DIP_INCONSISTENT
                notes.append(
                    f"inclination relative to gravity spans {dip_variation:.1f}° "
                    "within the window; the local field is being disturbed"
                )
    else:
        notes.append("no gravity estimate available, so inclination was not checked")

    weight = _weight_from(issues, good_fraction, magnitude_cv, median_magnitude)
    if weight == 0.0:
        notes.append("magnetic heading will not be used")
    elif weight < 1.0:
        notes.append(f"magnetic heading down-weighted to {weight:.2f}")

    return MagnetometerQuality(
        weight=weight,
        issues=issues,
        sample_count=len(mag),
        usable_count=usable_count,
        good_fraction=good_fraction,
        median_magnitude_ut=median_magnitude,
        magnitude_cv=magnitude_cv,
        max_direction_rate_rps=max_rate,
        dip_variation_deg=dip_variation,
        notes=notes,
    )


def _weight_from(
    issues: MagnetometerIssue,
    good_fraction: float,
    magnitude_cv: float | None,
    median_magnitude_ut: float,
) -> float:
    """Turn the individual findings into a single multiplier.

    Deliberately blunt. Two independent signs of disturbance drop the
    magnetometer entirely, because the failure mode that matters — a steady
    local field that biases heading by tens of degrees — is not something a
    carefully tuned partial weight would rescue.
    """
    if issues & MagnetometerIssue.UNUSABLE:
        return 0.0

    # A field this far outside the Earth's range is not the Earth's field with
    # some noise on it; it is a different field. Down-weighting it would keep a
    # heading that is simply wrong, so it is excluded on that single ground.
    if (
        median_magnitude_ut > 2.0 * EARTH_FIELD_MAX_UT
        or median_magnitude_ut < 0.5 * EARTH_FIELD_MIN_UT
    ):
        return 0.0

    disturbance_flags = [
        MagnetometerIssue.MAGNITUDE_OUT_OF_RANGE,
        MagnetometerIssue.ABRUPT_DIRECTION_CHANGE,
        MagnetometerIssue.UNSTABLE_MAGNITUDE,
        MagnetometerIssue.DIP_INCONSISTENT,
    ]
    triggered = sum(1 for flag in disturbance_flags if issues & flag)
    if triggered >= 2:
        return 0.0
    if triggered == 1:
        return 0.3

    # Clean window: scale gently with how many samples sat in the plausible
    # magnitude band, and penalise a wobbly magnitude even below the flag.
    weight = good_fraction
    if magnitude_cv is not None:
        weight *= max(0.0, 1.0 - magnitude_cv / 0.15)
    return float(min(1.0, max(0.0, weight)))
