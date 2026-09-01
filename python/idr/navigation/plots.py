"""Phase 4 diagnostic figures.

Each answers a question the validation report makes a claim about. Written to
``reports/navigation/plots/``; earlier phases' figures are untouched.

Two labelling rules, enforced by hand because matplotlib cannot enforce them:

* nothing here is captioned "ground truth". IO-VNBD documents none, and the
  comparison signal is a *reference* with its own known defects.
* every figure showing a real-data error says which lag it was computed at, and
  the drift figure shows the zero-lag and lag-corrected curves together. The gap
  between them is synchronization error, not navigation error, and a figure
  showing only the flattering one would misattribute it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if TYPE_CHECKING:
    from idr.navigation.diagnostics import (
        BiasSensitivityResult,
        DiagnosticRun,
        SegmentEvaluation,
    )
    from idr.navigation.synthetic import SyntheticTrajectory
    from idr.navigation.trajectory import Trajectory

FIGURE_DPI = 120
_BLUE = "#2a6f97"
_ORANGE = "#e07a5f"
_GREEN = "#3d8168"
_RED = "#d1495b"
_GREY = "#8d99ae"
_PURPLE = "#6a4c93"


def _finish(fig: plt.Figure, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=FIGURE_DPI)
    plt.close(fig)
    return output


def plot_synthetic_recovery(
    cases: dict[str, tuple[SyntheticTrajectory, Trajectory]], output: Path
) -> Path | None:
    """Position error against the analytical solution, per synthetic case.

    Log scale, because the interesting fact is the order of magnitude: these
    errors are at float64 rounding, and a linear axis would show four flat lines
    at zero and prove nothing.
    """
    if not cases:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for (name, (truth, propagated)), colour in zip(
        sorted(cases.items()), (_BLUE, _ORANGE, _GREEN, _RED, _PURPLE, _GREY), strict=False
    ):
        count = min(len(truth), len(propagated))
        error = np.linalg.norm(propagated.position_m[:count] - truth.position_nav_m[:count], axis=1)
        elapsed = propagated.analysis_time_s[:count] - propagated.analysis_time_s[0]
        ax.semilogy(elapsed, np.maximum(error, 1e-18), label=name, color=colour, linewidth=1.4)
    ax.set_title("Synthetic recovery: position error against the analytical solution")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Position error (m, log scale)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _finish(fig, output)


def plot_trajectory_xy(evaluation: SegmentEvaluation, output: Path) -> Path | None:
    """Propagated horizontal track against the GNSS reference track."""
    trajectory = evaluation.trajectory
    if trajectory is None or len(trajectory) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.plot(
        trajectory.position_m[:, 0],
        trajectory.position_m[:, 1],
        color=_BLUE,
        linewidth=1.5,
        label="Inertial baseline (unaided)",
    )
    ax.plot(
        trajectory.position_m[0, 0], trajectory.position_m[0, 1], "o", color=_GREEN, label="Start"
    )

    comparison = evaluation.comparison
    if comparison is not None and comparison.best_lag.yaw_offset_removed_deg is not None:
        ax.set_title(
            f"{evaluation.segment_id}: propagated track\n"
            f"(a {comparison.best_lag.yaw_offset_removed_deg:+.0f}° heading offset is "
            "unresolved and not removed here)",
            fontsize=10,
        )
    else:
        ax.set_title(f"{evaluation.segment_id}: propagated horizontal track", fontsize=11)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _finish(fig, output)


def plot_speed_comparison(evaluation: SegmentEvaluation, output: Path) -> Path | None:
    """Inertial speed against the reference, at zero lag and at the best lag."""
    trajectory = evaluation.trajectory
    comparison = evaluation.comparison
    if trajectory is None or comparison is None:
        return None
    from idr.navigation.reference_alignment import resample_onto

    reference = _reference_of(evaluation)
    if reference is None or reference.speed_mps is None:
        return None

    elapsed = trajectory.analysis_time_s - trajectory.analysis_time_s[0]
    masked = (
        reference.speed_mps
        if reference.speed_valid is None
        else np.where(reference.speed_valid, reference.speed_mps, np.nan)
    )
    zero = resample_onto(masked, reference.analysis_time_s, trajectory.analysis_time_s, 0.0)
    best = resample_onto(
        masked, reference.analysis_time_s, trajectory.analysis_time_s, comparison.lag.lag_s
    )

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(elapsed, trajectory.speed_mps, color=_BLUE, linewidth=1.5, label="Inertial baseline")
    ax.plot(elapsed, zero, color=_GREY, linewidth=1.2, label="Reference at zero lag")
    if comparison.lag.accepted:
        ax.plot(
            elapsed,
            best,
            color=_ORANGE,
            linewidth=1.2,
            linestyle="--",
            label=f"Reference read at t{comparison.lag.lag_s:+.1f}s",
        )
    ax.set_title(
        f"{evaluation.segment_id}: unaided inertial speed vs the {reference.speed_source} "
        "reference",
        fontsize=10,
    )
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _finish(fig, output)


def plot_error_growth(evaluation: SegmentEvaluation, output: Path) -> Path | None:
    """Speed and position error against elapsed time, both lags shown."""
    comparison = evaluation.comparison
    if comparison is None:
        return None
    zero, best = comparison.zero_lag, comparison.best_lag
    if zero.speed_error_mps is None and zero.horizontal_position_error_m is None:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    elapsed = zero.analysis_time_s - zero.analysis_time_s[0]

    if zero.speed_error_mps is not None:
        axes[0].plot(elapsed, zero.speed_error_mps, color=_GREY, linewidth=1.2, label="Zero lag")
        if comparison.lag.accepted and best.speed_error_mps is not None:
            axes[0].plot(
                elapsed,
                best.speed_error_mps,
                color=_ORANGE,
                linewidth=1.2,
                label=f"Lag {comparison.lag.lag_s:+.1f}s",
            )
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Speed error (m/s)")
    axes[0].set_title(
        f"{evaluation.segment_id}: unaided inertial error growth (reference, not ground truth)",
        fontsize=10,
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    if zero.horizontal_position_error_m is not None:
        axes[1].plot(
            elapsed, zero.horizontal_position_error_m, color=_GREY, linewidth=1.2, label="Zero lag"
        )
        if comparison.lag.accepted and best.horizontal_position_error_m is not None:
            axes[1].plot(
                elapsed,
                best.horizontal_position_error_m,
                color=_ORANGE,
                linewidth=1.2,
                label=f"Lag {comparison.lag.lag_s:+.1f}s",
            )
        if best.yaw_offset_removed_deg is not None:
            axes[1].text(
                0.02,
                0.92,
                f"a constant {best.yaw_offset_removed_deg:+.0f}° yaw offset was removed "
                "before comparing; this is a lower bound",
                transform=axes[1].transAxes,
                fontsize=7.5,
                color=_RED,
            )
    axes[1].set_ylabel("Horizontal position error (m)")
    axes[1].set_xlabel("Elapsed time (s)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    return _finish(fig, output)


def plot_acceleration_components(evaluation: SegmentEvaluation, output: Path) -> Path | None:
    """Gravity-compensated navigation acceleration, with its resting residual.

    The figure the gravity-sign claim rests on. During a detected stationary
    interval every component must sit at zero; a compensation error would show
    as a constant offset of several m/s² on the vertical channel and would be
    unmissable here.
    """
    trajectory = evaluation.trajectory
    if trajectory is None or len(trajectory) < 2:
        return None
    elapsed = trajectory.analysis_time_s - trajectory.analysis_time_s[0]
    acceleration = trajectory.acceleration_nav

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for index, (label, colour) in enumerate((("East", _BLUE), ("North", _GREEN), ("Up", _ORANGE))):
        ax.plot(elapsed, acceleration[:, index], color=colour, linewidth=1.0, label=label)

    from idr.navigation.state import NavigationFlag

    stationary = (trajectory.flags & int(NavigationFlag.STATIONARY)).astype(bool)
    if stationary.any():
        ax.fill_between(
            elapsed,
            *ax.get_ylim(),
            where=stationary,
            color=_GREY,
            alpha=0.18,
            label="Detected stationary",
        )
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_title(
        f"{evaluation.segment_id}: navigation-frame acceleration after gravity "
        "compensation (a = f + g)",
        fontsize=10,
    )
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=4)
    return _finish(fig, output)


def plot_dt_distribution(run: DiagnosticRun, output: Path) -> Path | None:
    """Sampling intervals actually encountered, against the integration limit."""
    from idr.navigation.integration import MAX_INTEGRATION_STEP_S

    intervals: list[float] = []
    for evaluation in run.evaluations:
        if evaluation.trajectory is None or len(evaluation.trajectory) < 2:
            continue
        steps = np.diff(evaluation.trajectory.analysis_time_s)
        intervals.extend(steps[np.isfinite(steps)].tolist())
    if not intervals:
        return None

    values = np.array(intervals)
    positive = values[values > 0.0]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if positive.size:
        edges = np.logspace(-4, 1.2, 80).tolist()
        ax.hist(positive, bins=edges, color=_BLUE, alpha=0.85)
        ax.set_xscale("log")
    ax.axvline(
        MAX_INTEGRATION_STEP_S,
        color=_RED,
        linestyle="--",
        label=f"Integration limit {MAX_INTEGRATION_STEP_S:g}s",
    )
    non_positive = int((values <= 0.0).sum())
    ax.set_title(
        "Sampling intervals encountered across the run "
        f"({non_positive:,} were ≤ 0 and were not integrated across)",
        fontsize=10,
    )
    ax.set_xlabel("Δt (s, log scale)")
    ax.set_ylabel("Steps")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _finish(fig, output)


def plot_drift_distribution(run: DiagnosticRun, output: Path) -> Path | None:
    """How fast the baseline leaves the reference, across every segment."""
    zero_rates: list[float] = []
    best_rates: list[float] = []
    for evaluation in run.evaluations:
        comparison = evaluation.comparison
        if comparison is None:
            continue
        if comparison.drift_zero_lag.position_drift_mps is not None:
            zero_rates.append(comparison.drift_zero_lag.position_drift_mps)
        if comparison.drift_best_lag.position_drift_mps is not None:
            best_rates.append(comparison.drift_best_lag.position_drift_mps)
    if not zero_rates and not best_rates:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    data = [values for values in (zero_rates, best_rates) if values]
    labels = [
        label
        for label, values in (("Zero lag", zero_rates), ("Lag corrected", best_rates))
        if values
    ]
    ax.boxplot(data, tick_labels=labels)
    ax.set_yscale("log")
    ax.set_title("Horizontal position drift rate of the unaided baseline, per segment", fontsize=10)
    ax.set_ylabel("Final horizontal error ÷ duration (m/s, log scale)")
    ax.grid(alpha=0.3, axis="y", which="both")
    return _finish(fig, output)


def plot_speed_error_by_status(run: DiagnosticRun, output: Path) -> Path | None:
    """Speed RMSE grouped by Phase 3 alignment outcome.

    Includes the segments where the alignment was unavailable, which is the
    point: §34 asks for the failure cases to be shown alongside the successes,
    not filtered out of the summary.
    """
    groups: dict[str, list[float]] = {}
    for evaluation in run.evaluations:
        comparison = evaluation.comparison
        if comparison is None:
            continue
        value = comparison.best_lag.summary()["speed_rmse_mps"]
        if value is None or not np.isfinite(value):
            continue
        groups.setdefault(str(evaluation.alignment_status), []).append(float(value))
    if not groups:
        return None

    names = sorted(groups, key=lambda name: -len(groups[name]))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.boxplot([groups[name] for name in names], tick_labels=[n.replace("_", "\n") for n in names])
    for index, name in enumerate(names, start=1):
        ax.text(
            index,
            ax.get_ylim()[1],
            f"n={len(groups[name])}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=_GREY,
        )
    ax.set_yscale("log")
    ax.set_title("Unaided speed RMSE by Phase 3 alignment outcome (lag corrected)", fontsize=10)
    ax.set_ylabel("Speed RMSE (m/s, log scale)")
    ax.grid(alpha=0.3, axis="y", which="both")
    return _finish(fig, output)


def plot_lag_effect(run: DiagnosticRun, output: Path) -> Path | None:
    """Speed RMSE at zero lag against the same at the corrected lag.

    Everything below the diagonal is apparent navigation error that is really
    residual synchronization error. Quantifying that split is §27's purpose.
    """
    zero: list[float] = []
    best: list[float] = []
    for evaluation in run.evaluations:
        comparison = evaluation.comparison
        if comparison is None or not comparison.lag.accepted:
            continue
        first = comparison.zero_lag.summary()["speed_rmse_mps"]
        second = comparison.best_lag.summary()["speed_rmse_mps"]
        if first is None or second is None:
            continue
        if np.isfinite(first) and np.isfinite(second):
            zero.append(float(first))
            best.append(float(second))
    if not zero:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(zero, best, s=22, color=_BLUE, alpha=0.7, edgecolor="none")
    limit = max(max(zero), max(best)) * 1.1
    floor = min(min(zero), min(best)) * 0.9
    ax.plot(
        [floor, limit], [floor, limit], color=_RED, linestyle="--", linewidth=1.0, label="No change"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        f"Effect of removing the residual lag on speed RMSE ({len(zero)} segments)", fontsize=10
    )
    ax.set_xlabel("Speed RMSE at zero lag (m/s)")
    ax.set_ylabel("Speed RMSE at the corrected lag (m/s)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _finish(fig, output)


def plot_bias_sensitivity(results: list[BiasSensitivityResult], output: Path) -> Path | None:
    """Trajectory divergence caused by a deliberate bias perturbation."""
    points: dict[str, list[tuple[float, float]]] = {}
    for result in results:
        for point in result.points:
            if not np.isfinite(point.divergence_from_baseline_m):
                continue
            kind = "Gyroscope (rad/s)" if point.axis.startswith("gyro") else "Accelerometer (m/s²)"
            points.setdefault(kind, []).append((point.magnitude, point.divergence_from_baseline_m))
    if not points:
        return None

    fig, axes = plt.subplots(1, len(points), figsize=(5.4 * len(points), 4.6), squeeze=False)
    for axis, (kind, values) in zip(axes[0], sorted(points.items()), strict=False):
        magnitudes = sorted({magnitude for magnitude, _ in values})
        grouped = [
            [divergence for magnitude, divergence in values if magnitude == target]
            for target in magnitudes
        ]
        axis.boxplot(grouped, tick_labels=[f"{value:g}" for value in magnitudes])
        # A log axis needs something positive on it. An all-zero column is a
        # real outcome (a perturbation the segment was too short to feel), so
        # the scale adapts rather than the data being nudged off zero.
        if any(value > 0.0 for column in grouped for value in column):
            axis.set_yscale("log")
        axis.set_title(f"{kind} perturbation", fontsize=10)
        axis.set_xlabel("Perturbation magnitude")
        axis.set_ylabel("Divergence from the unperturbed run (m, log)")
        axis.grid(alpha=0.3, axis="y", which="both")
    fig.suptitle("Sensitivity of the unaided baseline to a fixed IMU bias error", fontsize=11)
    return _finish(fig, output)


def plot_resting_residual(run: DiagnosticRun, output: Path) -> Path | None:
    """Compensated acceleration during detected rest — the gravity-sign evidence.

    At rest the true linear acceleration is zero, so whatever survives
    compensation is error. This figure is the real-data counterpart of the
    synthetic gravity-sign test: a sign fault would place the distribution at
    ~19.6 m/s² rather than near zero.
    """
    from idr.navigation.state import NavigationFlag

    magnitudes: list[float] = []
    verticals: list[float] = []
    for evaluation in run.evaluations:
        trajectory = evaluation.trajectory
        if trajectory is None:
            continue
        resting = (trajectory.flags & int(NavigationFlag.STATIONARY)).astype(bool)
        usable = resting & np.isfinite(trajectory.acceleration_nav).all(axis=1)
        if not usable.any():
            continue
        selected = trajectory.acceleration_nav[usable]
        magnitudes.append(float(np.linalg.norm(selected.mean(axis=0))))
        verticals.append(float(selected[:, 2].mean()))
    if not magnitudes:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].hist(magnitudes, bins=40, color=_BLUE, alpha=0.85)
    axes[0].set_title("‖mean a_nav‖ during detected rest", fontsize=10)
    axes[0].set_xlabel("Residual acceleration magnitude (m/s²)")
    axes[0].set_ylabel("Segments")
    axes[0].grid(alpha=0.3)
    axes[1].hist(verticals, bins=40, color=_ORANGE, alpha=0.85)
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Vertical component during rest (0 is correct)", fontsize=10)
    axes[1].set_xlabel("Residual vertical acceleration (m/s²)")
    axes[1].grid(alpha=0.3)
    fig.suptitle(
        f"Gravity compensation on real data, {len(magnitudes)} segments with a detected "
        "stationary interval",
        fontsize=11,
    )
    return _finish(fig, output)


def generate_run_figures(run: DiagnosticRun, directory: Path) -> list[Path]:
    """Every run-level figure. Returns the paths actually written."""
    directory.mkdir(parents=True, exist_ok=True)
    candidates = [
        (plot_dt_distribution, "01_dt_distribution.png"),
        (plot_resting_residual, "02_resting_residual.png"),
        (plot_drift_distribution, "03_drift_distribution.png"),
        (plot_speed_error_by_status, "04_error_by_alignment_status.png"),
        (plot_lag_effect, "05_lag_effect.png"),
    ]
    written: list[Path] = []
    for function, name in candidates:
        result = function(run, directory / name)
        if result is not None:
            written.append(result)
    if run.bias_sensitivity:
        result = plot_bias_sensitivity(run.bias_sensitivity, directory / "06_bias_sensitivity.png")
        if result is not None:
            written.append(result)
    return written


def generate_segment_figures(
    evaluation: SegmentEvaluation, directory: Path, prefix: str
) -> list[Path]:
    """Per-segment figures for one chosen example."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for function, suffix in (
        (plot_trajectory_xy, "trajectory"),
        (plot_speed_comparison, "speed"),
        (plot_error_growth, "error"),
        (plot_acceleration_components, "acceleration"),
    ):
        result = function(evaluation, directory / f"{prefix}_{suffix}.png")
        if result is not None:
            written.append(result)
    return written


def _reference_of(evaluation: SegmentEvaluation):  # type: ignore[no-untyped-def]
    """The reference track a figure needs, reloaded only if it was not retained.

    ``SegmentEvaluation`` deliberately does not hold the whole reference — the
    diagnostics run keeps 154 of these alive at once — so the plot re-derives it
    from the segment on disk. Reading is safe; nothing here writes.
    """
    from idr.navigation.diagnostics import prepare_segment

    inputs = prepare_segment(evaluation.path)
    return None if inputs is None else inputs.reference
