"""Diagnostic plots for the forensic report.

Each plot answers one question about the data and is labelled with units and
the session it came from, so a figure lifted out of the report still carries
its own context. Nothing decorative is produced.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Non-interactive backend: plots are written to disk, never displayed, so this
# must be selected before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from idr.dataset.scan import StreamScan  # noqa: E402
from idr.dataset.synchronization import SynchronizationReport  # noqa: E402
from idr.dataset.timestamps import TimestampReport  # noqa: E402

FIGURE_DPI = 120


def _label(relative_path: str) -> str:
    """Short, readable identifier for a plot title.

    The dataset's relative paths are long enough to overflow a figure title,
    so titles carry the filename (which uniquely identifies the stream and
    session) and the full path stays in the metadata artifacts.
    """
    return Path(relative_path).name


def _finish(fig: plt.Figure, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=FIGURE_DPI)
    plt.close(fig)
    return output


def plot_interval_distribution(
    timestamps: TimestampReport, scan: StreamScan, output: Path
) -> Path | None:
    """Histogram of inter-sample intervals — shows cadence and jitter."""
    if scan.time_seconds is None:
        return None
    values = scan.time_seconds[np.isfinite(scan.time_seconds)]
    if values.size < 3:
        return None
    deltas = np.diff(values)
    deltas = deltas[(deltas > 0) & (deltas < np.percentile(deltas, 99.5))]
    if deltas.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(deltas * 1000.0, bins=80, color="#2a6f97", edgecolor="none")
    if timestamps.delta_median_s:
        ax.axvline(
            timestamps.delta_median_s * 1000.0,
            color="#d1495b",
            linestyle="--",
            label=f"median {timestamps.delta_median_s * 1000:.2f} ms "
            f"({timestamps.estimated_rate_hz:.3f} Hz)",
        )
        ax.legend()
    ax.set_title(f"Sampling interval distribution — {_label(scan.relative_path)}")
    ax.set_xlabel("Interval between consecutive samples (ms)")
    ax.set_ylabel("Sample count")
    return _finish(fig, output)


def plot_inertial_signal(
    scan: StreamScan, roles: tuple[str, str, str], title: str, ylabel: str, output: Path
) -> Path | None:
    """Plot a 3-axis inertial signal against elapsed time.

    Re-reads the file because the scan retains only aggregates for these
    channels; this is the one place a second read is justified and it touches
    only the handful of sessions selected for plotting.
    """
    import pandas as pd

    columns = {
        role: column.raw_name
        for column in scan.schema.columns
        for role in roles
        if column.semantic_role == role
    }
    if len(columns) != 3 or scan.time_seconds is None:
        return None

    path = Path(scan.source_path) if scan.source_path else None
    if path is None or not path.exists():
        return None

    frame = pd.read_csv(
        path, encoding=scan.schema.encoding, usecols=list(columns.values()), low_memory=False
    )
    time = scan.time_seconds[: len(frame)]
    time = time - time[0]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for role, colour in zip(roles, ("#2a6f97", "#e07a5f", "#3d8168"), strict=True):
        series = pd.to_numeric(frame[columns[role]], errors="coerce").to_numpy(
            dtype="float64", na_value=np.nan
        )
        ax.plot(time, series[: len(time)], linewidth=0.6, color=colour, label=role)
    ax.set_title(f"{title} — {_label(scan.relative_path)}")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    return _finish(fig, output)


def plot_gnss_availability(scan: StreamScan, output: Path) -> Path | None:
    """Availability of a GNSS fix over the session timeline."""
    if scan.gnss_valid is None or scan.time_seconds is None:
        return None
    valid = scan.gnss_valid
    time = scan.time_seconds[: valid.size]
    if time.size == 0:
        return None
    time = time - time[0]

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.fill_between(time, 0, valid.astype(float), step="mid", color="#3d8168", linewidth=0)
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no fix", "fix"])
    ax.set_title(
        f"GNSS fix availability ({100 * valid.mean():.2f}% of samples)\n"
        f"{_label(scan.relative_path)}"
    )
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Fix state")
    return _finish(fig, output)


def plot_synchronization_timeline(
    reports: list[SynchronizationReport], output: Path, limit: int = 18
) -> Path | None:
    """Smartphone vs vehicle coverage on a shared time-of-day axis."""
    usable = [r for r in reports if r.smartphone_start_s is not None][:limit]
    if not usable:
        return None

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(usable) + 2.2))
    for row, report in enumerate(usable):
        assert report.smartphone_start_s is not None
        assert report.smartphone_end_s is not None
        assert report.vehicle_start_s is not None
        assert report.vehicle_end_s is not None
        ax.hlines(
            row + 0.16,
            report.smartphone_start_s / 3600.0,
            report.smartphone_end_s / 3600.0,
            color="#2a6f97",
            linewidth=5,
        )
        ax.hlines(
            row - 0.16,
            report.vehicle_start_s / 3600.0,
            report.vehicle_end_s / 3600.0,
            color="#e07a5f",
            linewidth=5,
        )
    ax.set_yticks(range(len(usable)))
    ax.set_yticklabels([r.session_id for r in usable], fontsize=8)
    ax.set_xlabel("Time of day (hours since local midnight)")
    ax.set_ylabel("Session")
    ax.set_title(
        "Smartphone (blue, upper) vs vehicle (orange, lower) coverage\n"
        "as recorded — offsets NOT corrected"
    )
    return _finish(fig, output)


def plot_trajectory(scan: StreamScan, output: Path) -> Path | None:
    """Geographic trajectory from the GNSS fixes of one session."""
    if scan.trajectory is None:
        return None
    latitude, longitude = scan.trajectory
    mask = np.isfinite(latitude) & np.isfinite(longitude) & ~((latitude == 0) & (longitude == 0))
    if mask.sum() < 2:
        return None

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(longitude[mask], latitude[mask], linewidth=1.0, color="#2a6f97")
    ax.scatter(
        longitude[mask][0], latitude[mask][0], color="#3d8168", s=45, label="start", zorder=3
    )
    ax.scatter(
        longitude[mask][-1], latitude[mask][-1], color="#d1495b", s=45, label="end", zorder=3
    )
    ax.set_title(f"GNSS trajectory — {_label(scan.relative_path)}")
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    return _finish(fig, output)


def plot_missingness_summary(
    rows: list[tuple[str, float]], output: Path, limit: int = 25
) -> Path | None:
    """Columns with the highest missing/unparseable percentage dataset-wide."""
    if not rows:
        return None
    top = sorted(rows, key=lambda item: item[1], reverse=True)[:limit]
    labels = [name for name, _ in top][::-1]
    values = [value for _, value in top][::-1]

    fig, ax = plt.subplots(figsize=(9, 0.32 * len(labels) + 2.0))
    ax.barh(labels, values, color="#e07a5f")
    ax.set_xlabel("Rows missing or unparseable (%)")
    ax.set_ylabel("Column")
    ax.set_title("Highest-missingness columns across the dataset")
    ax.tick_params(axis="y", labelsize=8)
    return _finish(fig, output)


def plot_speed_distribution(
    speeds: np.ndarray, label: str, output: Path, unit: str = "km/h"
) -> Path | None:
    """Distribution of recorded speed for one session."""
    finite = speeds[np.isfinite(speeds)]
    if finite.size < 3:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(finite, bins=60, color="#2a6f97", edgecolor="none")
    ax.axvline(
        float(finite.mean()),
        color="#d1495b",
        linestyle="--",
        label=f"mean {finite.mean():.1f} {unit}",
    )
    ax.set_title(f"Speed distribution — {_label(label)}")
    ax.set_xlabel(f"Speed ({unit})")
    ax.set_ylabel("Sample count")
    ax.legend()
    return _finish(fig, output)
