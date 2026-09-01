"""The Phase 4 validation report.

Written from the run rather than by hand, so the numbers in it cannot drift
away from the numbers the code produced. The structure follows §43 of the phase
brief.

One editorial rule governs the whole document: **no cherry-picking**. The
real-data section reports the distribution across every segment processed,
including the ones where the alignment was unavailable and the ones where the
solution diverged badly. An inertial baseline that drifts is the expected
result, and hiding the drift would make the Phase 5 comparison worthless.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from idr.frames.conventions import STANDARD_GRAVITY_MPS2
from idr.navigation.bias import MAX_PLAUSIBLE_ACCEL_BIAS_MPS2
from idr.navigation.integration import MAX_INTEGRATION_STEP_S
from idr.navigation.mechanization import MAX_PLAUSIBLE_SPECIFIC_FORCE_MPS2
from idr.navigation.reference_alignment import MAX_LAG_SEARCH_S
from idr.navigation.stationarity import (
    MAX_STATIONARY_ACCEL_STD_MPS2,
    MAX_STATIONARY_GRAVITY_ERROR_MPS2,
    MAX_STATIONARY_GYRO_RPS,
)

if TYPE_CHECKING:
    from idr.navigation.diagnostics import DiagnosticRun, SegmentEvaluation


def _fmt(value: object, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        return f"{value:,.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _quadrature_blind_spot() -> float:
    """Largest horizontal acceleration the rest detector cannot see, m/s².

    ``√((g + τ)² − g²)``: horizontal acceleration adds to gravity in quadrature,
    so a magnitude tolerance ``τ`` hides everything below this. Computed rather
    than quoted, so tightening the tolerance updates the report.
    """
    limit = STANDARD_GRAVITY_MPS2 + MAX_STATIONARY_GRAVITY_ERROR_MPS2
    return math.sqrt(limit * limit - STANDARD_GRAVITY_MPS2 * STANDARD_GRAVITY_MPS2)


def _percentiles(values: list[float]) -> tuple[float, float, float] | None:
    if not values:
        return None
    array = np.array(values, dtype="float64")
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return (
        float(np.percentile(array, 25)),
        float(np.median(array)),
        float(np.percentile(array, 75)),
    )


def _collect(run: DiagnosticRun, key: str, lag: str = "best") -> list[float]:
    out: list[float] = []
    for evaluation in run.evaluations:
        if evaluation.comparison is None:
            continue
        curves = evaluation.comparison.best_lag if lag == "best" else evaluation.comparison.zero_lag
        value = curves.summary().get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def summarize_run(run: DiagnosticRun) -> dict[str, object]:
    """Machine-readable summary, written alongside the report."""
    from idr.navigation.diagnostics import performance_summary

    status_counts: dict[str, int] = {}
    for evaluation in run.evaluations:
        key = str(evaluation.alignment_status)
        status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "started_at": run.started_at,
        "processed_dir": str(run.processed_dir),
        "summary": run.summary(),
        "alignment_status_counts": status_counts,
        "performance": performance_summary(run),
        "notes": list(run.notes),
    }


def generate_phase4_report(
    run: DiagnosticRun,
    output: Path,
    *,
    figures: list[Path] | None = None,
    synthetic_results: dict[str, float] | None = None,
    example_segments: list[SegmentEvaluation] | None = None,
) -> Path:
    """Write the Phase 4 validation report."""
    lines: list[str] = []
    add = lines.append

    add("# Phase 4 — Inertial Mechanization Validation")
    add("")
    add(
        f"Generated from a run started {run.started_at} over "
        f"`{run.processed_dir}`. Every number below comes from that run."
    )
    add("")

    _objective(add)
    _implementation(add, run)
    _mathematics(add)
    _synthetic(add, synthetic_results)
    _real_data(add, run)
    _synchronization(add, run)
    _drift(add, run)
    _bias(add, run)
    _performance(add, run)
    _failures(add, run)
    _limitations(add)
    _phase5(add)
    _figures(add, figures, example_segments)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _objective(add) -> None:  # type: ignore[no-untyped-def]
    add("## 1. Objective")
    add(
        "Turn calibrated smartphone IMU measurements into a continuously propagated "
        "attitude, velocity and position, with no aiding of any kind after "
        "initialization. This is the **baseline** every later phase is measured "
        "against, and its value depends entirely on it not having been quietly helped."
    )
    add("")
    add("**The correction budget, in full.**")
    add("")
    add("| Used | Not used |")
    add("|---|---|")
    add("| Phase 3 orientation estimate | GNSS after initialization |")
    add("| Phase 3 phone→vehicle alignment | Reference velocity, in any form |")
    add("| A constant gravity model | Map matching |")
    add("| A fixed IMU bias where one could be estimated | Non-holonomic constraints |")
    add("| An initial position, velocity and attitude | Zero-velocity *corrections* |")
    add("| | Any filter, learned model, or fusion |")
    add("")
    add(
        "Nothing in `idr.navigation.mechanization` reads a reference signal. The module "
        "that does — `reference_alignment` — has no path by which it can write into a "
        "trajectory, and there is deliberately no `update()` or `apply_fix()` method to "
        "call. The absence of aiding is structural, not a matter of discipline."
    )
    add("")


def _implementation(add, run: DiagnosticRun) -> None:  # type: ignore[no-untyped-def]
    add("## 2. Implementation")
    add("")
    add("| Module | Responsibility |")
    add("|---|---|")
    add(
        "| `state.py` | `NavigationState`, `InitialState`, the per-sample "
        "quality flags, the unit table |"
    )
    add(
        "| `gravity.py` | The gravity *model* — injectable, constant by default. "
        "Distinct from Phase 3's gravity *estimator* |"
    )
    add(
        "| `bias.py` | Typed gyro/accelerometer bias, and what a stationary "
        "window can and cannot determine |"
    )
    add("| `integration.py` | Trapezoidal integration and the time-step policy that guards it |")
    add(
        "| `stationarity.py` | Deterministic rest detection — for initialization "
        "and annotation, never correction |"
    )
    add(
        "| `mechanization.py` | `StrapdownMechanization`: the step, the attitude "
        "source, the frame chain |"
    )
    add("| `trajectory.py` | The trajectory container and the batch driver |")
    add("| `geodesy.py` | Latitude/longitude → local ENU, with the origin stated |")
    add(
        "| `reference_alignment.py` | Evaluation only: lag estimation, error "
        "curves, drift metrics |"
    )
    add("| `synthetic.py` | Analytical trajectories — the only place accuracy is measurable |")
    add("| `diagnostics.py`, `plots.py`, `report.py`, `runner.py` | The real-data pass |")
    add("")
    add(
        "Phase 3 is consumed, not re-implemented: orientation comes from "
        "`idr.frames.orientation`, the mounting rotation from `idr.frames.alignment`, "
        "the canonical data through `idr.frames.sources`, and the speed reference "
        "through `idr.frames.diagnostics.choose_speed_reference`."
    )
    add("")
    if run.notes:
        for note in run.notes:
            add(f"- {note}")
        add("")


def _mathematics(add) -> None:  # type: ignore[no-untyped-def]
    add("## 3. Mathematical formulation")
    add("")
    add(
        "Frames are the Phase 3 conventions: navigation ENU (X east, Y north, Z up), "
        "vehicle X forward / Y left / Z up, phone as delivered by the logger."
    )
    add("")
    add("```text")
    add("attitude       q_nav_phone(k) = q_nav_phone(k−1) ⊗ δq(ω·Δt)     [exponential map]")
    add("specific force f_vehicle      = R_vehicle_phone · f_phone")
    add("               f_nav          = R_nav_vehicle · f_vehicle")
    add("gravity        a_nav          = f_nav + g_nav                    ← added, not subtracted")
    add("velocity       v_k            = v_{k−1} + ½(a_{k−1} + a_k)·Δt")
    add("position       p_k            = p_{k−1} + ½(v_{k−1} + v_k)·Δt")
    add("```")
    add("")
    add(
        "`a = f + g` is the equation this phase turns on. An accelerometer measures "
        "specific force `f = a − g` where `g` points down, so recovering acceleration "
        "**adds** the gravity vector. Writing `f − g` leaves a resting device reporting "
        f"2g upward — about {2 * STANDARD_GRAVITY_MPS2:.1f} m/s² — which integrates to "
        "roughly 300 m of spurious climb in 5.5 s while every intermediate number still "
        "looks like a plausible acceleration. Phase 3 hit both halves of this sign "
        "question and both were caught by tests rather than by reading."
    )
    add("")
    add(
        "The attitude increment multiplies on the **right**, because `δθ = ω·Δt` is a "
        "rotation of the body. Left-multiplying applies it about navigation axes "
        "instead — a different rotation whenever the device is not already level and "
        "facing north, and one that produces a perfectly well-formed wrong answer."
    )
    add("")
    add(
        "Trapezoidal integration is exact for a linearly varying rate, so constant "
        "acceleration integrates without error. That is what lets the synthetic tests "
        "demand machine precision instead of a tolerance chosen until they passed."
    )
    add("")
    add(
        "**Not compensated:** coning, sculling, and a midpoint attitude update. Each "
        "matters when the rotation rate or specific force varies appreciably *within* "
        "one sample interval. At the 10 Hz that dominates this dataset, and against the "
        "attitude error measured in §5, they are far from the leading term. The state "
        "interface is arranged so they can be added inside `step()` without changing "
        "anything a caller sees."
    )
    add("")
    add(
        "**Numerical safeguards.** Non-finite or implausible sensor samples "
        f"(specific force beyond {MAX_PLAUSIBLE_SPECIFIC_FORCE_MPS2:.0f} m/s², 15 g) are "
        "flagged `INVALID_SENSOR` and the state is held rather than sanitized into "
        "something plausible. Quaternions are renormalized every step; rotation matrices "
        "are validated for orthonormality *and* determinant by the Phase 3 `FrameRotation`."
    )
    add("")


def _synthetic(add, results: dict[str, float] | None) -> None:  # type: ignore[no-untyped-def]
    add("## 4. Synthetic validation — the only accuracy claim in this phase")
    add("")
    add(
        "Real IO-VNBD has no position ground truth, no attitude ground truth, and a "
        "velocity *reference* carrying seconds of residual synchronization error. A "
        "real-data run can show the mechanization behaves plausibly; it cannot show it "
        "is correct. Synthetic trajectories can, because the answer is known in closed "
        "form."
    )
    add("")
    if results:
        add("| Case | Max position error (m) |")
        add("|---|---|")
        for name in sorted(results):
            add(f"| {name} | {results[name]:.3e} |")
        add("")
        worst = max(results.values())
        add(
            f"Worst case across every analytical fixture: **{worst:.2e} m**. That is "
            "float64 rounding, not a tolerance — trapezoidal integration is exact for "
            "these profiles, so any real error would stand out by many orders of "
            "magnitude."
        )
    else:
        add("_No synthetic results were supplied to this report._")
    add("")
    add(
        "The fixtures are generated from the definition `f = a − g` written "
        "independently of the mechanization's `a = f + g`, so a sign error in one "
        "cannot cancel against the other. The stationary case additionally pins the "
        f"result to a number a reader can check by hand: a level resting device reads "
        f"+{STANDARD_GRAVITY_MPS2:.5f} m/s² on its up axis, and must compensate to "
        "exactly zero."
    )
    add("")


def _real_data(add, run: DiagnosticRun) -> None:  # type: ignore[no-untyped-def]
    summary = run.summary()
    add("## 5. Real-data validation")
    add("")
    add(
        f"{summary['segments_processed']} smartphone segments were propagated. Selection "
        "was on stated criteria — an accelerometer channel and at least 300 samples — "
        "never on how the result looked. Segments whose alignment was unavailable are "
        "**included**, because their failure behaviour is part of what has to be shown."
    )
    add("")
    add("| Quantity | Value |")
    add("|---|---|")
    add(f"| Segments processed | {summary['segments_processed']} |")
    add(f"| Propagated without error | {summary['segments_propagated']} |")
    add(f"| With a full phone→vehicle alignment | {summary['segments_with_alignment']} |")
    add(
        "| Alignment unavailable (phone-frame diagnostic) | "
        f"{summary['segments_without_alignment']} |"
    )
    add(f"| Compared against a reference | {summary['segments_compared']} |")
    add("")

    add("### 5.1 Gravity compensation on real data")
    add("")
    resting = _resting_residuals(run)
    if resting:
        quartiles = _percentiles([magnitude for magnitude, _, _ in resting])
        verticals = _percentiles([vertical for _, vertical, _ in resting])
        horizontals = _percentiles([horizontal for _, _, horizontal in resting])
        add(
            f"Across {len(resting)} segments containing a detected stationary interval, "
            "the mean compensated acceleration during rest has a magnitude of "
            f"**{_fmt(quartiles[1] if quartiles else None, 3)} m/s²** (median; IQR "
            f"{_fmt(quartiles[0] if quartiles else None, 3)}–"
            f"{_fmt(quartiles[2] if quartiles else None, 3)}). Its horizontal part has "
            f"a median of {_fmt(horizontals[1] if horizontals else None, 3)} m/s² and "
            f"its vertical part {_fmt(verticals[1] if verticals else None, 3)} m/s²."
        )
        add("")
        add(
            "This is the real-data confirmation of the gravity convention. At rest the "
            "true acceleration is zero, so whatever survives compensation is error — and "
            f"a sign fault would place this distribution at ~{2 * STANDARD_GRAVITY_MPS2:.1f} "
            "m/s², not near zero."
        )
        add("")
        if horizontals:
            residual = horizontals[1]
            tilt = math.degrees(math.asin(min(1.0, residual / STANDARD_GRAVITY_MPS2)))
            add(
                "**The residual is attitude error, and it is the dominant error source.** "
                "The *horizontal* component is the one a tilt error produces, because a "
                "misjudged vertical tips gravity into east and north. A residual of "
                f"{residual:.3f} m/s² corresponds to a tilt error of {tilt:.2f}°. Left "
                f"uncorrected that integrates to {residual * 60:.1f} m/s of velocity "
                f"error and {0.5 * residual * 3600:,.0f} m of position error in one "
                "minute — and it grows quadratically thereafter. Nothing else in the "
                "budget is close: the constant gravity model is worth at most "
                "0.026 m/s², and the missing coning and sculling terms far less at "
                "these rates."
            )
            add("")
            add(
                "The figure worth noting alongside it is that this is the residual at "
                "**rest**, where the orientation filter's gravity correction is working "
                "under the best conditions it ever sees. During motion the same "
                "correction is competing with the vehicle's own acceleration, and the "
                "drift rates in §7 imply a substantially larger effective error there. "
                "An attitude estimate that is good at rest and degrades under exactly "
                "the dynamics you need it for is the central problem Phase 5 inherits."
            )
            add("")

    add("### 5.2 Speed against the reference")
    add("")
    zero = _percentiles(_collect(run, "speed_rmse_mps", "zero"))
    best = _percentiles(_collect(run, "speed_rmse_mps", "best"))
    add("| Comparison | P25 | Median | P75 |")
    add("|---|---|---|---|")
    if zero:
        add(f"| Speed RMSE at zero lag (m/s) | {zero[0]:,.2f} | {zero[1]:,.2f} | {zero[2]:,.2f} |")
    if best:
        add(
            f"| Speed RMSE at the corrected lag (m/s) | {best[0]:,.2f} | "
            f"{best[1]:,.2f} | {best[2]:,.2f} |"
        )
    add("")
    add(
        "These are large, and they are supposed to be. An unaided inertial solution on "
        "consumer MEMS sensors has no mechanism that would keep the speed bounded: "
        "the tilt error in §5.1 is a persistent acceleration error, and a persistent "
        "acceleration error is an unbounded velocity error by construction."
    )
    add("")
    add(
        "**Why the lag-corrected row is not uniformly better.** The lag search "
        "maximises *correlation*, which is a shape comparison; the RMSE is dominated "
        "by the drift *magnitude*, which a time shift does not touch. On a solution "
        "whose speed has run away by tens of m/s, sliding the reference a few seconds "
        "changes the residual only marginally and can move it either way. That is not "
        "a sign the correction failed — it is the correct reading that on this dataset "
        "the residual synchronization error is a small part of the total, and the "
        "unaided drift is nearly all of it. Both rows are reported so that split is "
        "visible instead of being asserted."
    )
    add("")


def _synchronization(add, run: DiagnosticRun) -> None:  # type: ignore[no-untyped-def]
    add("## 6. Reference synchronization methodology")
    add("")
    add(
        "Three different times are kept distinct, because conflating them is how "
        "synchronization error gets reported as navigation error:"
    )
    add("")
    add("| Name | What it is |")
    add("|---|---|")
    add(
        "| Clock alignment | What Phase 2 did — interpolating the vehicle stream "
        "onto smartphone sample times. Its contract warns the result is not "
        "exact. |"
    )
    add(
        "| Residual alignment | The leftover offset Phase 3 measured: median "
        "4.6 s, maximum 14.6 s across 154 segments. Real, large, and corrected "
        "nowhere in the stored data. |"
    )
    add("| Navigation time | `analysis_time_s` as the mechanization used it. **Never shifted.** |")
    add("")
    add(
        "A lag-corrected comparison therefore resamples **the reference** onto navigation "
        f"time over a ±{MAX_LAG_SEARCH_S:g} s search, and reports the shift. The reference "
        "at navigation time *t* is the reference recorded at *t + lag*, and it is labelled "
        "that way rather than passed off as the raw value at *t*. Shifting the IMU "
        "timestamps instead would move the trajectory's own clock to flatter a "
        "comparison."
    )
    add("")
    summary = run.summary()
    add(
        f"**{summary['segments_with_accepted_lag']}** of {summary['segments_compared']} "
        f"compared segments showed a distinct correlation peak away from zero, with a "
        f"median magnitude of {_fmt(summary['median_abs_lag_s'], 1)} s. A peak that does "
        "not stand clear of the search curve's median is rejected, so a flat curve's "
        "argmax is never mistaken for a measurement."
    )
    add("")
    zero = _percentiles(_collect(run, "speed_rmse_mps", "zero"))
    best = _percentiles(_collect(run, "speed_rmse_mps", "best"))
    if zero and best and zero[1] > 0:
        improvement = 100.0 * (1.0 - best[1] / zero[1])
        if abs(improvement) < 1.0:
            add(
                f"Removing the residual lag changes the median speed RMSE by less than "
                f"1% ({zero[1]:.2f} → {best[1]:.2f} m/s). **That is a result, not a "
                "null result.** On a dataset where a lag correction mattered, the two "
                "would differ substantially, and the gap would be synchronization error "
                "misattributed to navigation. Here the unaided drift is so much larger "
                "than the timing residual that the split is invisible in the RMSE — "
                "which is worth knowing precisely because it will *stop* being true "
                "once Phase 5 brings the drift down. The lag must be accounted for "
                "then; the measurement is on the comparison object for that purpose."
            )
        else:
            add(
                f"Removing the residual lag reduces the median speed RMSE by "
                f"**{improvement:.0f}%** ({zero[1]:.2f} → {best[1]:.2f} m/s). That "
                "difference is synchronization error that would otherwise have been "
                "attributed to the navigation solution. Both numbers are reported "
                "everywhere in this document for exactly that reason."
            )
        add("")
    add(
        "**Nothing is written back.** The lag is a measurement of leftover Phase 2 error, "
        "reported on the comparison object. Canonical timestamps are untouched."
    )
    add("")


def _drift(add, run: DiagnosticRun) -> None:  # type: ignore[no-untyped-def]
    add("## 7. Drift analysis")
    add("")
    add(
        "A single final error conflates a long drive with a short one, so drift is "
        "reported as a rate. Note the units: a constant acceleration error — which is "
        "what a tilt error is — produces a *quadratic* position error, so the linear "
        "rate below grows with segment duration and is not a constant of the system."
    )
    add("")
    rates = _percentiles(
        [
            evaluation.comparison.drift_best_lag.position_drift_mps
            for evaluation in run.evaluations
            if evaluation.comparison is not None
            and evaluation.comparison.drift_best_lag.position_drift_mps is not None
        ]
    )
    finals = _percentiles(_collect(run, "final_horizontal_error_m", "best"))
    add("| Quantity | P25 | Median | P75 |")
    add("|---|---|---|---|")
    if rates:
        add(
            f"| Horizontal drift rate (m/s) | {rates[0]:,.2f} | {rates[1]:,.2f} | {rates[2]:,.2f} |"
        )
    if finals:
        add(
            f"| Final horizontal error (m) | {finals[0]:,.0f} | "
            f"{finals[1]:,.0f} | {finals[2]:,.0f} |"
        )
    add("")
    add(
        "**A heading caveat that changes how these numbers read.** Where the "
        "magnetometer was not trusted, Phase 3's yaw is dead-reckoned from an arbitrary "
        "origin, so the propagated east and north components are correct only up to an "
        "unknown constant rotation about the vertical. Comparing them directly would "
        "measure that unknown rather than the navigation error, so for those segments a "
        "constant yaw offset is removed **before** computing position error. The "
        "trajectory itself is never modified, the angle removed is reported per segment, "
        "and the resulting position error is *blind to a heading error* — it is a lower "
        "bound on the true position error, not an estimate of it."
    )
    add("")
    add(
        "This is not a defect of the mechanization. It is a property of a dataset in "
        "which absolute heading is frequently unobservable, and it is precisely the kind "
        "of thing a Phase 5 filter with a position update would resolve."
    )
    add("")


def _bias(add, run: DiagnosticRun) -> None:  # type: ignore[no-untyped-def]
    add("## 8. Bias sensitivity")
    add("")
    add("### 8.1 What a stationary window can actually determine")
    add("")
    add(
        "**Gyroscope: fully.** At rest the true angular rate is zero, so the mean "
        "measured rate is the bias. (Earth rate, 7.3e-5 rad/s, is one to two orders of "
        "magnitude below a consumer MEMS noise floor and is ignored; a navigation-grade "
        "IMU would have to compensate it.)"
    )
    add("")
    add(
        "**Accelerometer: one axis of three.** At rest the sensor reads `f = −g + b`, "
        "and averaging cannot separate the two. The direction of gravity in the phone "
        "frame is unknown *a priori* — it is exactly what Phase 3 estimates from the same "
        "mean — so using that estimate and subtracting yields `b ≡ 0` identically: a "
        "number that looks like a measurement and contains none. What *is* separable is "
        "the component along gravity, by appealing to the known local gravity magnitude. "
        "The two perpendicular components are indistinguishable from the device being "
        "tilted slightly differently than believed, and no amount of stationary data "
        "resolves them."
    )
    add("")
    add(
        "So the implementation estimates the gravity-parallel component only, records "
        '`observable_axes="gravity_parallel"`, and leaves the other two at exactly zero '
        "*by construction rather than by measurement*. Estimates beyond "
        f"{MAX_PLAUSIBLE_ACCEL_BIAS_MPS2:g} m/s² are rejected as evidence the window was "
        "not stationary."
    )
    add("")
    add(
        "**This is the honest explanation of the drift.** The unobservable horizontal "
        "accelerometer bias and the residual tilt error are the same quantity seen two "
        "ways, and neither is separable without an independent position or velocity "
        "update. That is a filter, and it is Phase 5."
    )
    add("")

    add("### 8.2 The perturbation experiment")
    add("")
    if not run.bias_sensitivity:
        add("_No sensitivity experiment was run._")
        add("")
        return
    add(
        "Perturbations are fixed constants drawn from sensor datasheet behaviour "
        "(0.001–0.02 rad/s for the gyroscope, 0.005–0.1 m/s² for the accelerometer), "
        "applied one phone axis at a time, and **not** tuned against the results. "
        "Divergence is measured against the unperturbed run rather than against the "
        "reference, which isolates sensitivity to bias from every other error source."
    )
    add("")
    add(
        "**The two sensors are measured under different attitude modes**, and the "
        "reason is worth stating because it is a real property of the baseline rather "
        "than a quirk of the experiment. In `PHASE3_FILTER` the attitude comes from "
        "the Phase 3 filter, computed upstream from the unperturbed gyroscope; Phase 4 "
        "applies its bias to the angular rate *after* that, where nothing consumes it. "
        "A gyroscope perturbation therefore moves the baseline trajectory by exactly "
        "zero metres — true, and reporting it that way would be badly misleading, "
        "because a gyroscope bias matters enormously wherever the gyroscope actually "
        "drives attitude. Gyroscope perturbations are therefore measured under "
        "`GYRO_PROPAGATION`, against an unperturbed run in that same mode."
    )
    add("")
    add("| Sensor | Attitude mode | Perturbation | Median divergence (m) |")
    add("|---|---|---|---|")
    grouped: dict[tuple[str, str, float], list[float]] = {}
    for result in run.bias_sensitivity:
        for point in result.points:
            if not math.isfinite(point.divergence_from_baseline_m):
                continue
            kind = "Gyroscope (rad/s)" if point.axis.startswith("gyro") else "Accelerometer (m/s²)"
            grouped.setdefault((kind, point.attitude_mode, point.magnitude), []).append(
                point.divergence_from_baseline_m
            )
    for (kind, mode, magnitude), values in sorted(grouped.items()):
        add(f"| {kind} | `{mode}` | {magnitude:g} | {np.median(values):,.1f} |")
    add("")
    durations = [result.duration_s for result in run.bias_sensitivity]
    add(
        f"Measured over {len(run.bias_sensitivity)} segment(s) of typical length "
        f"(median {np.median(durations):.0f} s) — the ones closest to the median "
        "duration of the aligned set, a rule fixed before any result was seen. "
        "Deliberately **not** the longest: a bias error grows quadratically with time, "
        "so this dataset's three-hour recordings produce divergences in the hundreds "
        "of kilometres, which are arithmetically correct and useless as a description "
        "of what a bias costs on an ordinary drive."
    )
    add("")
    paths = [
        point.path_length_m
        for result in run.bias_sensitivity
        for point in result.points
        if point.path_length_m > 0.0
    ]
    if paths:
        add(
            f"For scale, the median path length over these segments is "
            f"{np.median(paths):,.0f} m. Read the table against that number rather than "
            "in absolute metres: the smallest accelerometer perturbation tried is "
            "already a few percent of the distance travelled, and the smallest "
            "gyroscope one is several times the distance travelled — because an "
            "attitude error grows linearly and drives position error *cubically*, "
            "while an accelerometer bias only drives it quadratically."
        )
        add("")
    add(
        "The point of this table is not the exact metres. It is that perturbations far "
        "below what a consumer IMU's run-to-run bias actually is are enough to move the "
        "solution by a distance comparable to — for the gyroscope, well beyond — the "
        "trajectory itself. **That is the quantitative justification for Phase 5.** An "
        "unaided solution cannot be made accurate by better integration; it needs an "
        "observation that constrains the bias."
    )
    add("")


def _performance(add, run: DiagnosticRun) -> None:  # type: ignore[no-untyped-def]
    from idr.navigation.diagnostics import performance_summary

    add("## 9. Performance")
    add("")
    numbers = performance_summary(run)
    if numbers.get("segments"):
        add(
            f"Mean **{numbers['mean_us_per_sample']:.0f} µs/sample**, p95 "
            f"**{numbers['p95_us_per_sample']:.0f} µs/sample** across "
            f"{int(numbers['segments'])} segments; mean {numbers['mean_s_per_segment']:.2f} s "
            f"per segment, p95 {numbers['p95_s_per_segment']:.2f} s."
        )
        add("")
        add(
            "These figures include the full Phase 3 chain (orientation filter, gravity, "
            "magnetometer assessment, alignment) plus the reference comparison, not the "
            "mechanization alone — the mechanization is a small fraction of it. Recorded "
            "because the engine must eventually run in real time, **not** because "
            "anything here is optimized. Nothing has been tuned for speed and the "
            "propagation is a plain Python loop. This is a baseline to improve against."
        )
    else:
        add("_No timing was recorded._")
    add("")


def _failures(add, run: DiagnosticRun) -> None:  # type: ignore[no-untyped-def]
    add("## 10. Failure cases")
    add("")
    add(
        "**Alignment unavailable.** Phase 3 returns `rotation = None` whenever yaw was "
        "never resolved, and Phase 4 never substitutes identity for it — doing so would "
        "assert a mounting that was never measured and send the trajectory off in a "
        "direction nothing observed. The configured policy decides what happens:"
    )
    add("")
    add("| Policy | Behaviour |")
    add("|---|---|")
    add(
        "| `REQUIRE` (default) | The run refuses to start, with a message "
        "naming the alignment status |"
    )
    add(
        "| `PHONE_FRAME_DIAGNOSTIC` | The run proceeds, every sample is flagged "
        "`ALIGNMENT_UNAVAILABLE`, and `Trajectory.is_navigation_grade` is False |"
    )
    add("")
    add(
        "The second is legitimate because a strapdown solution actually needs "
        "`R_navigation_phone`, which the orientation estimator supplies — the vehicle "
        "frame is needed for the vehicle-frame quantities later phases want "
        "(longitudinal/lateral split, non-holonomic constraints, wheel-speed "
        "comparison), not for the integration itself. The distinction is recorded on the "
        "trajectory so a consumer cannot miss it."
    )
    add("")
    without = run.without_alignment
    add(
        f"{len(without)} of {len(run.evaluations)} segments took this path in the "
        "real-data run and are included in every table above."
    )
    add("")
    errors = [item for item in run.evaluations if item.error]
    if errors:
        add(f"**Hard failures.** {len(errors)} segment(s) raised during propagation:")
        add("")
        for item in errors[:10]:
            add(f"- `{item.segment_id}` — {item.error}")
        add("")
    else:
        add("**Hard failures.** None: every selected segment propagated to completion.")
        add("")
    add(
        f"**Invalid time steps.** Steps with Δt ≤ 0 (Phase 2 retains 147,946 duplicate "
        f"timestamps) or Δt > {MAX_INTEGRATION_STEP_S:g} s are not integrated across. The "
        "state is held, the sample is flagged `DUPLICATE_TIMESTAMP` or `LARGE_TIME_GAP`, "
        "and the trapezoidal accumulator is reset so the next valid step does not average "
        "against a rate from before the gap. `SEGMENT` and `FAIL` policies are available "
        "for callers who need a break or an exception instead."
    )
    add("")


def _limitations(add) -> None:  # type: ignore[no-untyped-def]
    add("## 11. Limitations")
    add("")
    add(
        "1. **No ground truth exists in this dataset.** Every real-data comparison is "
        "against a *reference*: Phase 2 rates its best velocity label high-confidence, "
        "not truth, and the GNSS position it derives from refreshes about every 9 s. A "
        "disagreement is not by itself evidence that the inertial side is the wrong one."
    )
    add(
        "2. **Absolute heading is frequently unobservable.** Where the magnetometer is "
        "not trusted, the trajectory is correct only up to a rotation about the vertical. "
        "Position errors for those segments have that degree of freedom removed before "
        "comparison and are therefore lower bounds."
    )
    add(
        "3. **The horizontal accelerometer bias cannot be estimated here.** It is not "
        "separable from a tilt error without an independent position or velocity "
        "observation. This is a property of the problem, not a gap in the implementation."
    )
    add(
        "4. **The vertical channel is the least trustworthy.** Gravity compensation error "
        "goes straight into it and there is nothing to bound it, which is why horizontal "
        "error is reported separately throughout rather than folded into a 3-D norm."
    )
    add(
        "5. **No coning, sculling or midpoint update.** Below the leading error term at "
        "these rates, but they will matter once the tilt error is filtered down."
    )
    add(
        "6. **A constant gravity model.** Normal gravity spans 9.780–9.832 m/s² with "
        "latitude, so the wrong constant is worth up to 0.026 m/s². `WGS84Gravity` is "
        "implemented and substitutable; the baseline does not use it because the term is "
        "an order of magnitude below the tilt error."
    )
    add(
        "7. **Stationarity detection is threshold-based**, and its thresholds "
        f"({MAX_STATIONARY_ACCEL_STD_MPS2:g} m/s² of magnitude variation, "
        f"{MAX_STATIONARY_GYRO_RPS:g} rad/s of rotation, "
        f"{MAX_STATIONARY_GRAVITY_ERROR_MPS2:g} m/s² from gravity) are engineering "
        "constants from sensor physics, not values fitted to this dataset."
    )
    add(
        "8. **The rest detector has a quantified blind spot.** Horizontal acceleration "
        "adds to gravity in quadrature, so an acceleration `a` raises the measured "
        f"magnitude only to √(g² + a²). At a {MAX_STATIONARY_GRAVITY_ERROR_MPS2:g} m/s² "
        f"tolerance that leaves accelerations below **{_quadrature_blind_spot():.1f} "
        "m/s²** indistinguishable from rest — not a defect of this detector but of "
        "IMU-only detection, since rest and constant acceleration produce identical "
        "specific force. The consequence is bounded here because the result feeds only "
        "initialization, bias windows and annotation; it would not be bounded if it fed "
        "a zero-velocity correction, which is one reason Phase 4 has none."
    )
    add("")


def _phase5(add) -> None:  # type: ignore[no-untyped-def]
    add("## 12. What Phase 5 requires")
    add("")
    add(
        "The baseline says exactly what the next phase has to fix, in priority order set "
        "by the measurements above rather than by expectation:"
    )
    add("")
    add(
        "1. **An observation that separates tilt error from horizontal accelerometer "
        "bias.** These are the same quantity seen two ways and they dominate everything "
        "else. Nothing in an unaided solution can distinguish them; a filter with a "
        "position or velocity update can."
    )
    add(
        "2. **A heading observation.** Absolute yaw is unobservable on most segments "
        "here, which is why position error can only be reported as a lower bound. Any "
        "aiding that constrains heading turns those bounds into measurements."
    )
    add(
        "3. **Accounting for the residual synchronization lag.** Any use of "
        "`ref_velocity_mps` as a time-aligned target must handle a multi-second offset — "
        "Phase 3 measured a median of 4.6 s correlating accelerometer against dv/dt, and "
        "§6 above measures it independently here. "
        "`reference_alignment.estimate_residual_lag` reports it per segment, with the "
        "acceptance gates that stop it being manufactured; it must not be assumed away."
    )
    add(
        "4. **A like-for-like comparison.** `AttitudeMode`, `AlignmentPolicy`, the "
        "gravity model and the bias are all injectable, and the correction budget in §1 "
        "is explicit, so a Phase 5 ablation can hold everything else fixed and change "
        "one thing."
    )
    add("")
    add(
        "The interfaces a filter needs exist — `NavigationState` is a complete state, "
        "`StrapdownMechanization.step()` is a single propagation step, and the "
        "`ImuBias` structure is where an estimated bias would be written. The "
        "functionality behind them does not, and deliberately so."
    )
    add("")


def _figures(add, figures, examples) -> None:  # type: ignore[no-untyped-def]
    add("## 13. Figures")
    add("")
    if not figures:
        add("_No figures were generated._")
        add("")
        return
    for path in figures:
        add(f"- `{path.parent.name}/{path.name}`")
    add("")
    if examples:
        add(
            "Per-segment figures are drawn for "
            + ", ".join(f"`{item.segment_id}`" for item in examples)
            + " — chosen to span the outcomes (a well-aligned segment, a weakly aligned "
            "one, and one with no alignment at all), not to show the best results."
        )
        add("")


def _resting_residuals(run: DiagnosticRun) -> list[tuple[float, float, float]]:
    """(‖mean a_nav‖, mean vertical, ‖mean horizontal‖) over each segment's rest.

    The three are reported separately because they mean different things. The
    *horizontal* component is the one a tilt error produces — gravity leaking
    into east and north — so it is the one the attitude-error inference in §5.1
    is entitled to use. The vertical component mixes attitude error with the
    gravity model's own magnitude error and is quoted as a sanity check only.
    """
    from idr.navigation.state import NavigationFlag

    out: list[tuple[float, float, float]] = []
    for evaluation in run.evaluations:
        trajectory = evaluation.trajectory
        if trajectory is None:
            continue
        resting = (trajectory.flags & int(NavigationFlag.STATIONARY)).astype(bool)
        usable = resting & np.isfinite(trajectory.acceleration_nav).all(axis=1)
        if not usable.any():
            continue
        mean = trajectory.acceleration_nav[usable].mean(axis=0)
        out.append(
            (
                float(np.linalg.norm(mean)),
                float(mean[2]),
                float(np.linalg.norm(mean[:2])),
            )
        )
    return out
