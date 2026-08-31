"""Phase 2 diagnostic plots.

Each figure answers a question the Phase 2 report makes a claim about, so a
reader can check the claim rather than take it on trust. Written to
``reports/io_vnbd/phase2_plots/`` — Phase 1's figures are left untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if TYPE_CHECKING:
    from idr.pipeline.runner import PipelineResult

FIGURE_DPI = 120
_BLUE = "#2a6f97"
_ORANGE = "#e07a5f"
_GREEN = "#3d8168"
_RED = "#d1495b"


def _finish(fig: plt.Figure, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=FIGURE_DPI)
    plt.close(fig)
    return output


def plot_segment_sizes(result: PipelineResult, output: Path) -> Path | None:
    """How the dataset breaks into logical trips."""
    durations = [
        s.duration_s for s in result.segments if np.isfinite(s.duration_s) and s.duration_s > 0
    ]
    if len(durations) < 2:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(np.array(durations) / 60.0, bins=60, color=_BLUE, edgecolor="none")
    ax.set_title(f"Logical segment durations after trip splitting (n={len(durations)})")
    ax.set_xlabel("Segment duration (minutes)")
    ax.set_ylabel("Segment count")
    return _finish(fig, output)


def plot_multi_segment_files(result: PipelineResult, output: Path, limit: int = 20) -> Path | None:
    """Files that turned out to hold more than one trip."""
    multi = sorted(
        [t for t in result.timing if t.segment_count > 1],
        key=lambda t: -t.segment_count,
    )[:limit]
    if not multi:
        return None
    names = [Path(t.source_file).name for t in multi][::-1]
    counts = [t.segment_count for t in multi][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.33 * len(names) + 2.0))
    ax.barh(names, counts, color=_ORANGE)
    ax.set_title("Source files split into multiple logical trips")
    ax.set_xlabel("Segments produced")
    ax.set_ylabel("Source file")
    ax.tick_params(axis="y", labelsize=8)
    return _finish(fig, output)


def plot_sync_offsets(result: PipelineResult, output: Path) -> Path | None:
    """Distribution of estimated smartphone/vehicle offsets."""
    offsets = [
        s.estimated_offset_s
        for s in result.sync
        if s.estimated_offset_s is not None and np.isfinite(s.estimated_offset_s)
    ]
    if len(offsets) < 2:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(np.array(offsets) / 3600.0, bins=60, color=_BLUE, edgecolor="none")
    ax.set_title(f"Estimated S/V time offset per pair (n={len(offsets)})")
    ax.set_xlabel("Offset (hours; 0 = aligned, 1 = one-hour clock difference)")
    ax.set_ylabel("Pair count")
    return _finish(fig, output)


def plot_sync_confidence(result: PipelineResult, output: Path) -> Path | None:
    """Correlation achieved by the speed-based aligner."""
    values = [
        (s.correlation, s.status)
        for s in result.sync
        if s.correlation is not None and np.isfinite(s.correlation)
    ]
    if len(values) < 2:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist([v for v, _ in values], bins=40, color=_GREEN, edgecolor="none")
    ax.axvline(0.5, color=_RED, linestyle="--", label="confirmation threshold (0.50)")
    ax.set_title("Speed cross-correlation at the estimated offset")
    ax.set_xlabel("Pearson correlation between smartphone and vehicle speed")
    ax.set_ylabel("Pair count")
    ax.legend()
    return _finish(fig, output)


def plot_gnss_freshness(result: PipelineResult, output: Path) -> Path | None:
    """Measured position-update interval, smartphone vs vehicle."""
    phone = [
        r.median_update_interval_s
        for r in result.freshness
        if r.stream == "smartphone" and r.median_update_interval_s
    ]
    vehicle = [
        r.median_update_interval_s
        for r in result.freshness
        if r.stream == "vehicle" and r.median_update_interval_s
    ]
    if not phone and not vehicle:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = list(np.logspace(-2, 2, 60))
    if phone:
        ax.hist(phone, bins=bins, alpha=0.75, color=_BLUE, label=f"smartphone (n={len(phone)})")
    if vehicle:
        ax.hist(vehicle, bins=bins, alpha=0.75, color=_ORANGE, label=f"vehicle (n={len(vehicle)})")
    ax.set_xscale("log")
    ax.set_title("Median GNSS position-update interval — a fix on every row is not a fresh fix")
    ax.set_xlabel("Seconds between actual position changes (log scale)")
    ax.set_ylabel("Stream count")
    ax.legend()
    return _finish(fig, output)


def plot_label_coverage(result: PipelineResult, output: Path) -> Path | None:
    """Reference-label coverage by source tier."""
    if not result.labels:
        return None
    by_source: dict[str, list[float]] = {}
    for label in result.labels:
        by_source.setdefault(label.label_source, []).append(label.coverage_fraction)
    names = sorted(by_source)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([by_source[n] for n in names], tick_labels=[n.replace("_", "\n") for n in names])
    ax.set_title("Reference-velocity label coverage per segment, by source tier")
    ax.set_ylabel("Fraction of rows with a usable label")
    ax.tick_params(axis="x", labelsize=7)
    return _finish(fig, output)


def plot_split_balance(result: PipelineResult, output: Path) -> Path | None:
    """Row counts per split, to show the grouped split is still balanced."""
    if not result.split_summary or not result.split_summary.rows:
        return None
    names = sorted(result.split_summary.rows)
    rows = [result.split_summary.rows[n] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, rows, color=[_BLUE, _ORANGE, _GREEN][: len(names)])
    total = sum(rows) or 1
    for bar, value in zip(bars, rows, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{100 * value / total:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Rows per split (grouped by payload hash and session — no leakage)")
    ax.set_ylabel("Canonical rows")
    return _finish(fig, output)


def plot_sampling_rates(result: PipelineResult, output: Path) -> Path | None:
    """Measured rates preserved through canonicalization (no forced 10 Hz)."""
    phone: list[float] = []
    vehicle: list[float] = []
    for timing, read in zip(result.timing, result.reads, strict=False):
        if timing.measured_rate_hz:
            (phone if read.stream == "smartphone" else vehicle).append(timing.measured_rate_hz)
    if not phone and not vehicle:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = list(np.logspace(0, 3.2, 70))
    if phone:
        ax.hist(phone, bins=bins, alpha=0.75, color=_BLUE, label=f"smartphone (n={len(phone)})")
    if vehicle:
        ax.hist(vehicle, bins=bins, alpha=0.75, color=_ORANGE, label=f"vehicle (n={len(vehicle)})")
    ax.set_xscale("log")
    ax.axvline(10.0, color=_RED, linestyle="--", label="nominal 10 Hz")
    ax.set_title("Measured sampling rate per stream, preserved (not resampled)")
    ax.set_xlabel("Measured rate (Hz, log scale)")
    ax.set_ylabel("Stream count")
    ax.legend()
    return _finish(fig, output)


def generate_phase2_plots(result: PipelineResult, plots_dir: Path) -> list[Path]:
    """Produce every Phase 2 figure the run supports."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    for name, builder in (
        ("01_sampling_rates.png", plot_sampling_rates),
        ("02_segment_durations.png", plot_segment_sizes),
        ("03_multi_segment_files.png", plot_multi_segment_files),
        ("04_sync_offsets.png", plot_sync_offsets),
        ("05_sync_confidence.png", plot_sync_confidence),
        ("06_gnss_freshness.png", plot_gnss_freshness),
        ("07_label_coverage.png", plot_label_coverage),
        ("08_split_balance.png", plot_split_balance),
    ):
        path = builder(result, plots_dir / name)
        if path is not None:
            produced.append(path)
    return produced
