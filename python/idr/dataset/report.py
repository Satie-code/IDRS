"""Representative-session selection and diagnostic plot generation.

Sessions chosen for plotting are picked from measured evidence (GNSS gaps,
motion dynamics, synchronization findings), and each selection records the
reason it was made, so the figures in the report are defensible rather than
arbitrary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from idr.dataset.plots import (
    plot_gnss_availability,
    plot_inertial_signal,
    plot_interval_distribution,
    plot_missingness_summary,
    plot_speed_distribution,
    plot_synchronization_timeline,
    plot_trajectory,
)
from idr.dataset.scan import load_role_series
from idr.dataset.timestamps import analyze_timestamps
from idr.logging import get_logger

if TYPE_CHECKING:
    from idr.dataset.inspect import InspectionResult

logger = get_logger(__name__)


@dataclass(frozen=True)
class RepresentativeSession:
    """One session selected for detailed reporting, with the reason why."""

    category: str
    session_id: str
    relative_path: str
    reason: str


def select_representative_sessions(result: InspectionResult) -> list[RepresentativeSession]:
    """Pick sessions that illustrate distinct, evidenced dataset conditions."""
    selections: list[RepresentativeSession] = []
    taken: set[str] = set()

    quality_by_path = {q.relative_path: q for q in result.quality}
    outage_by_path = {o.relative_path: o for o in result.outage_reports}

    def add(category: str, path: str, reason: str) -> None:
        if path in taken or path not in result.scans:
            return
        taken.add(path)
        session_id = quality_by_path[path].session_id if path in quality_by_path else "unknown"
        selections.append(RepresentativeSession(category, session_id, path, reason))

    smartphone = [q for q in result.quality if q.stream == "smartphone" and q.row_count > 0]

    # 1. Clean session: full GNSS, stable sampling, most rows.
    clean = [
        q
        for q in smartphone
        if q.sampling_class == "stable"
        and q.gnss_availability >= 0.999
        and q.range_finding_count == 0
    ]
    if clean:
        best = max(clean, key=lambda q: q.row_count)
        add(
            "clean",
            best.relative_path,
            f"stable {best.measured_rate_hz:.2f} Hz sampling, "
            f"{100 * best.gnss_availability:.1f}% GNSS availability, no range findings, "
            f"{best.row_count:,} rows",
        )

    # 2. GNSS-gap session: the largest measured loss of fix.
    gapped = [o for o in outage_by_path.values() if o.outage_count > 0]
    if gapped:
        worst = max(gapped, key=lambda o: o.total_outage_s)
        add(
            "gnss_gap",
            worst.relative_path,
            f"{worst.outage_count} observed GNSS gap(s) totalling {worst.total_outage_s:.1f}s "
            f"(longest {worst.longest_outage_s:.1f}s); availability "
            f"{100 * worst.availability_fraction:.1f}%",
        )

    # 3. Difficult motion: most out-of-band range findings.
    dynamic = [q for q in smartphone if q.range_finding_count > 0]
    if dynamic:
        worst_dynamic = max(dynamic, key=lambda q: q.range_finding_count)
        add(
            "difficult_motion",
            worst_dynamic.relative_path,
            f"{worst_dynamic.range_finding_count} range finding(s) — highest in the dataset, "
            f"indicating strong dynamics or sensor artifacts",
        )

    # 4. Synchronized pair: longest verified overlap.
    paired = [s for s in result.synchronization if s.aligned_overlap_s]
    if paired:
        longest = max(paired, key=lambda s: s.aligned_overlap_s or 0.0)
        add(
            "synchronized_pair",
            longest.smartphone_file,
            f"paired S/V session with {longest.aligned_overlap_s:.0f}s aligned overlap; "
            f"{longest.interpretation}",
        )

    # 5. Irregular timing, if any exists.
    irregular = [q for q in smartphone if q.sampling_class in ("irregular", "severely_irregular")]
    if irregular:
        worst_timing = min(irregular, key=lambda q: q.timestamp_quality)
        add(
            "irregular_sampling",
            worst_timing.relative_path,
            f"sampling classified {worst_timing.sampling_class}; "
            f"timestamp quality {worst_timing.timestamp_quality:.3f}",
        )

    return selections


def generate_plots(result: InspectionResult, plots_dir: Path) -> list[Path]:
    """Produce every diagnostic figure the available data supports."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    def keep(path: Path | None) -> None:
        if path is not None:
            produced.append(path)

    selections = select_representative_sessions(result)
    by_category = {s.category: s for s in selections}

    primary = by_category.get("clean") or (selections[0] if selections else None)

    if primary is not None:
        scan = result.scans[primary.relative_path]
        timestamps = analyze_timestamps(scan)
        keep(
            plot_interval_distribution(timestamps, scan, plots_dir / "01_interval_distribution.png")
        )
        keep(
            plot_inertial_signal(
                scan,
                ("accel_x", "accel_y", "accel_z"),
                "Accelerometer",
                "Acceleration (m/s²)",
                plots_dir / "02_accelerometer.png",
            )
        )
        keep(
            plot_inertial_signal(
                scan,
                ("gyro_yaw", "gyro_pitch", "gyro_roll"),
                "Gyroscope",
                "Angular rate (rad/s)",
                plots_dir / "03_gyroscope.png",
            )
        )
        keep(plot_trajectory(scan, plots_dir / "06_trajectory.png"))

        speeds = load_role_series(scan, "gnss_speed")
        if speeds is not None:
            keep(
                plot_speed_distribution(
                    speeds, primary.relative_path, plots_dir / "08_speed_distribution.png"
                )
            )

    # GNSS availability: prefer a session that actually has gaps.
    gnss_choice = by_category.get("gnss_gap") or primary
    if gnss_choice is not None:
        keep(
            plot_gnss_availability(
                result.scans[gnss_choice.relative_path], plots_dir / "04_gnss_availability.png"
            )
        )

    keep(
        plot_synchronization_timeline(
            result.synchronization, plots_dir / "05_synchronization_timeline.png"
        )
    )

    # Missingness across every column of every scanned file.
    aggregated: dict[str, list[float]] = {}
    for record in result.missingness:
        share = record.missing_percentage
        if record.total_rows:
            share = max(share, 100.0 * record.non_numeric_rows / record.total_rows)
        aggregated.setdefault(record.column, []).append(share)
    rows = [(name, float(np.mean(values))) for name, values in aggregated.items()]
    keep(plot_missingness_summary(rows, plots_dir / "07_missingness_summary.png"))

    return produced
