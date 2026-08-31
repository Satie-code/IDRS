"""Phase 3 diagnostic figures.

Each figure answers a question the validation report makes a claim about, so a
reader can check the claim instead of taking it on trust. Written to
``reports/sensor_alignment/plots/``; earlier phases' figures are untouched.

Every figure that shows an alignment says *estimated* in its title. The true
phone→vehicle rotation for IO-VNBD is not documented anywhere, so nothing here
can be labelled ground truth without inventing a claim.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if TYPE_CHECKING:
    from idr.frames.diagnostics import DiagnosticRun, SegmentDiagnostic

FIGURE_DPI = 120
_BLUE = "#2a6f97"
_ORANGE = "#e07a5f"
_GREEN = "#3d8168"
_RED = "#d1495b"
_GREY = "#8d99ae"


def _finish(fig: plt.Figure, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=FIGURE_DPI)
    plt.close(fig)
    return output


def plot_gravity_confidence(run: DiagnosticRun, output: Path) -> Path | None:
    """How well the vertical could be pinned down, by method."""
    by_method: dict[str, list[float]] = {}
    for segment in run.segments:
        by_method.setdefault(str(segment.gravity.method), []).append(segment.gravity.confidence)
    usable = {name: values for name, values in by_method.items() if values}
    if not usable:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = sorted(usable)
    ax.boxplot([usable[name] for name in names], tick_labels=[n.replace("_", "\n") for n in names])
    ax.set_title("Estimated gravity confidence per segment, by method")
    ax.set_ylabel("Confidence")
    ax.set_ylim(-0.05, 1.05)
    return _finish(fig, output)


def plot_device_tilt(run: DiagnosticRun, output: Path) -> Path | None:
    """Angle between measured down and the phone's own −Z axis.

    A description of how the devices were actually held, not a claim that the
    phone frame means what Android says it means.
    """
    tilts = [
        tilt
        for segment in run.segments
        if (tilt := segment.gravity.tilt_from_phone_z_deg()) is not None
    ]
    if len(tilts) < 2:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(tilts, bins=36, range=(0.0, 180.0), color=_BLUE, edgecolor="none")
    ax.axvline(0.0, color=_GREEN, linestyle="--", label="flat, screen up")
    ax.axvline(90.0, color=_ORANGE, linestyle="--", label="on edge")
    ax.set_title(f"Estimated device tilt from the phone −Z axis (n={len(tilts)} segments)")
    ax.set_xlabel("Angle between measured gravity and phone −Z (degrees)")
    ax.set_ylabel("Segments")
    ax.legend()
    return _finish(fig, output)


def plot_calibration_outcomes(run: DiagnosticRun, output: Path) -> Path | None:
    """What the alignment engine concluded, and how often."""
    counts = run.status_counts()
    if not counts:
        return None
    names = list(counts)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [_GREEN if name == "SUCCESS" else _ORANGE for name in names]
    bars = ax.bar([n.replace("_", "\n") for n in names], [counts[n] for n in names], color=colors)
    total = sum(counts.values()) or 1
    for bar, name in zip(bars, names, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{100 * counts[name] / total:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Calibration outcome per segment — a refusal is a result, not a failure")
    ax.set_ylabel("Segments")
    ax.tick_params(axis="x", labelsize=8)
    return _finish(fig, output)


def plot_forward_correlation(run: DiagnosticRun, output: Path) -> Path | None:
    """Evidence behind each recovered heading."""
    values = [
        segment.alignment.quality.forward_correlation
        for segment in run.segments
        if segment.alignment is not None
        and segment.alignment.quality.forward_correlation is not None
    ]
    if len(values) < 2:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(values, bins=30, range=(-1.0, 1.0), color=_GREEN, edgecolor="none")
    ax.axvline(0.3, color=_RED, linestyle="--", label="acceptance threshold (0.30)")
    ax.set_title("Correlation between predicted and observed longitudinal acceleration")
    ax.set_xlabel("Pearson correlation at the estimated forward axis")
    ax.set_ylabel("Segments")
    ax.legend()
    return _finish(fig, output)


def plot_residual_lag(run: DiagnosticRun, output: Path) -> Path | None:
    """Residual synchronization error Phase 3 had to remove.

    Phase 2 measured a per-pair offset and deliberately left both streams as
    recorded; this is what remains between the phone's accelerometer and the
    interpolated reference velocity.
    """
    lags = [
        segment.forward_lag_s
        for segment in run.segments
        if segment.forward_lag_s is not None and segment.forward_lag_s != 0.0
    ]
    if len(lags) < 2:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(lags, bins=40, color=_ORANGE, edgecolor="none")
    ax.axvline(0.0, color=_GREY, linestyle="--", label="perfect synchronization")
    ax.axvline(
        float(np.median(lags)),
        color=_RED,
        linestyle="-",
        label=f"median {np.median(lags):+.1f}s",
    )
    ax.set_title(f"Residual phone/vehicle lag measured by Phase 3 (n={len(lags)})")
    ax.set_xlabel("Shift applied to the speed reference (s)")
    ax.set_ylabel("Segments")
    ax.legend()
    return _finish(fig, output)


def plot_magnetometer_trust(run: DiagnosticRun, output: Path) -> Path | None:
    """How often the magnetometer could be used at all."""
    weights = [segment.magnetometer.weight for segment in run.segments]
    if len(weights) < 2:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(weights, bins=22, range=(0.0, 1.05), color=_BLUE, edgecolor="none")
    ax.set_title("Magnetometer trust weight per segment (0 = excluded from heading)")
    ax.set_xlabel("Weight applied to the magnetic heading correction")
    ax.set_ylabel("Segments")
    return _finish(fig, output)


def plot_sampling_rates(run: DiagnosticRun, output: Path) -> Path | None:
    """A reminder, drawn from the data, that nothing here is 10 Hz by fiat."""
    rates = [
        segment.timing.measured_rate_hz
        for segment in run.segments
        if segment.timing.measured_rate_hz
    ]
    if len(rates) < 2:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(rates, bins=list(np.logspace(0, 3.2, 60)), color=_BLUE, edgecolor="none")
    ax.set_xscale("log")
    ax.axvline(10.0, color=_RED, linestyle="--", label="nominal 10 Hz")
    ax.set_title("Measured sampling rate of the segments analysed")
    ax.set_xlabel("Rate (Hz, log scale)")
    ax.set_ylabel("Segments")
    ax.legend()
    return _finish(fig, output)


def plot_orientation_example(segment: SegmentDiagnostic, output: Path) -> Path | None:
    """Estimated orientation over one segment, with its axis support stated."""
    track = segment.orientation
    if track is None or len(track) < 10:
        return None
    angles = track.euler_deg()
    elapsed = track.analysis_time_s - track.analysis_time_s[0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for index, (name, color) in enumerate((("roll", _BLUE), ("pitch", _ORANGE), ("yaw", _GREEN))):
        support = track.at(0).axis_support()[name]
        axes[0].plot(
            elapsed, angles[:, index], color=color, linewidth=1.0, label=f"{name} ({support})"
        )
    axes[0].set_ylabel("Degrees")
    axes[0].set_title(
        f"Estimated orientation — {segment.segment_id} "
        "(estimate, not ground truth: no reference attitude exists for this dataset)"
    )
    axes[0].legend(fontsize=8)

    axes[1].plot(elapsed, track.confidence, color=_GREY, linewidth=1.2)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("Confidence")
    axes[1].set_xlabel("Seconds from the start of the segment (analysis clock)")
    return _finish(fig, output)


def plot_frame_comparison(segment: SegmentDiagnostic, output: Path) -> Path | None:
    """Phone-frame against estimated vehicle-frame acceleration.

    The point of the figure: in the vehicle frame the longitudinal axis should
    carry the accelerating and braking, and it should look different from the
    lateral axis. In the phone frame the same motion is smeared across all
    three axes by the unknown mounting.
    """
    alignment = segment.alignment
    if alignment is None or alignment.rotation is None or segment.phone_accel is None:
        return None
    phone = segment.phone_accel
    if len(phone) < 10:
        return None
    vehicle = alignment.rotation.apply_series(phone)
    elapsed = phone.analysis_time_s - phone.analysis_time_s[0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True)
    for index, (name, color) in enumerate((("x", _BLUE), ("y", _ORANGE), ("z", _GREEN))):
        axes[0].plot(elapsed, phone.samples[:, index], color=color, linewidth=0.7, label=name)
    axes[0].set_title(f"Phone-frame acceleration — {segment.segment_id}")
    axes[0].set_ylabel("m/s²")
    axes[0].legend(fontsize=8, ncol=3)

    for index, (name, color) in enumerate((("forward", _BLUE), ("left", _ORANGE), ("up", _GREEN))):
        axes[1].plot(elapsed, vehicle.samples[:, index], color=color, linewidth=0.7, label=name)
    roll, pitch, yaw = alignment.rotation.euler_deg()
    axes[1].set_title(
        f"Estimated vehicle-frame acceleration (R_vehicle_phone: roll {roll:.0f}°, "
        f"pitch {pitch:.0f}°, yaw {yaw:.0f}°; confidence {alignment.confidence:.2f})"
    )
    axes[1].set_ylabel("m/s²")
    axes[1].set_xlabel("Seconds from the start of the segment (analysis clock)")
    axes[1].legend(fontsize=8, ncol=3)
    return _finish(fig, output)


def plot_synthetic_recovery(results: list[tuple[float, float]], output: Path) -> Path | None:
    """Recovery error against applied rotation, from the synthetic sweep.

    This is the only figure in the set that *can* show accuracy, because it is
    the only one where the true rotation is known.
    """
    if len(results) < 2:
        return None
    angles = [angle for angle, _ in results]
    errors = [error for _, error in results]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(angles, errors, marker="o", color=_GREEN, linewidth=1.5)
    ax.set_title("Synthetic alignment recovery: error against applied rotation")
    ax.set_xlabel("Applied phone→vehicle rotation (degrees)")
    ax.set_ylabel("Recovery error (degrees)")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    return _finish(fig, output)


def generate_phase3_plots(
    run: DiagnosticRun,
    plots_dir: Path,
    *,
    synthetic_results: list[tuple[float, float]] | None = None,
) -> list[Path]:
    """Produce every Phase 3 figure the run supports."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    for name, builder in (
        ("01_sampling_rates.png", plot_sampling_rates),
        ("02_gravity_confidence.png", plot_gravity_confidence),
        ("03_device_tilt.png", plot_device_tilt),
        ("04_magnetometer_trust.png", plot_magnetometer_trust),
        ("05_calibration_outcomes.png", plot_calibration_outcomes),
        ("06_forward_correlation.png", plot_forward_correlation),
        ("07_residual_lag.png", plot_residual_lag),
    ):
        path = builder(run, plots_dir / name)
        if path is not None:
            produced.append(path)

    if synthetic_results:
        path = plot_synthetic_recovery(synthetic_results, plots_dir / "08_synthetic_recovery.png")
        if path is not None:
            produced.append(path)

    # Two per-segment illustrations, chosen deterministically: the highest and
    # the lowest confidence among the successful alignments, so the reader sees
    # a good case and a marginal one rather than a hand-picked best.
    successes = sorted(
        run.succeeded(), key=lambda s: s.alignment.confidence if s.alignment else 0.0
    )
    examples = []
    if successes:
        examples.append(("best", successes[-1]))
        if len(successes) > 1:
            examples.append(("marginal", successes[0]))
    for label, segment in examples:
        for suffix, segment_builder in (
            ("orientation", plot_orientation_example),
            ("frames", plot_frame_comparison),
        ):
            path = segment_builder(segment, plots_dir / f"09_{label}_{suffix}.png")
            if path is not None:
                produced.append(path)
    return produced


def sweep_recovery_errors() -> list[tuple[float, float]]:
    """Run the synthetic rotation sweep and return (angle, error) pairs.

    Kept here rather than in a test so the report can quote measured numbers
    from the same code path a reader can rerun.
    """
    from idr.frames import quaternion as quat
    from idr.frames.alignment import estimate_alignment, linear_accel_for_alignment
    from idr.frames.gravity import estimate_gravity
    from idr.frames.synthetic import synthetic_drive

    axis = np.array([0.3, 0.4, 0.866])
    results: list[tuple[float, float]] = []
    for angle_deg in (0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0):
        truth = quat.from_axis_angle(axis, math.radians(angle_deg))
        drive = synthetic_drive(q_vehicle_phone=truth, noise_mps2=0.1)
        gravity = estimate_gravity(
            drive.accel_phone, gravity_channel=drive.gravity_phone, gyro=drive.gyro_phone
        )
        estimate = estimate_alignment(
            gravity=gravity,
            linear_accel=linear_accel_for_alignment(drive.accel_phone, gravity),
            speed_mps=drive.speed_mps,
            gyro=drive.gyro_phone,
            duration_s=float(drive.analysis_time_s[-1]),
        )
        if estimate.rotation is None:
            continue
        error = math.degrees(quat.angular_distance(estimate.rotation.quaternion, truth))
        results.append((angle_deg, error))
    return results
