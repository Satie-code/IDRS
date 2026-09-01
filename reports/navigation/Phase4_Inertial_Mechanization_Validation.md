# Phase 4 — Inertial Mechanization Validation

Generated from a run started 2026-08-31T17:38:43 over `data\processed\io_vnbd`. Every number below comes from that run.

## 1. Objective
Turn calibrated smartphone IMU measurements into a continuously propagated attitude, velocity and position, with no aiding of any kind after initialization. This is the **baseline** every later phase is measured against, and its value depends entirely on it not having been quietly helped.

**The correction budget, in full.**

| Used | Not used |
|---|---|
| Phase 3 orientation estimate | GNSS after initialization |
| Phase 3 phone→vehicle alignment | Reference velocity, in any form |
| A constant gravity model | Map matching |
| A fixed IMU bias where one could be estimated | Non-holonomic constraints |
| An initial position, velocity and attitude | Zero-velocity *corrections* |
| | Any filter, learned model, or fusion |

Nothing in `idr.navigation.mechanization` reads a reference signal. The module that does — `reference_alignment` — has no path by which it can write into a trajectory, and there is deliberately no `update()` or `apply_fix()` method to call. The absence of aiding is structural, not a matter of discipline.

## 2. Implementation

| Module | Responsibility |
|---|---|
| `state.py` | `NavigationState`, `InitialState`, the per-sample quality flags, the unit table |
| `gravity.py` | The gravity *model* — injectable, constant by default. Distinct from Phase 3's gravity *estimator* |
| `bias.py` | Typed gyro/accelerometer bias, and what a stationary window can and cannot determine |
| `integration.py` | Trapezoidal integration and the time-step policy that guards it |
| `stationarity.py` | Deterministic rest detection — for initialization and annotation, never correction |
| `mechanization.py` | `StrapdownMechanization`: the step, the attitude source, the frame chain |
| `trajectory.py` | The trajectory container and the batch driver |
| `geodesy.py` | Latitude/longitude → local ENU, with the origin stated |
| `reference_alignment.py` | Evaluation only: lag estimation, error curves, drift metrics |
| `synthetic.py` | Analytical trajectories — the only place accuracy is measurable |
| `diagnostics.py`, `plots.py`, `report.py`, `runner.py` | The real-data pass |

Phase 3 is consumed, not re-implemented: orientation comes from `idr.frames.orientation`, the mounting rotation from `idr.frames.alignment`, the canonical data through `idr.frames.sources`, and the speed reference through `idr.frames.diagnostics.choose_speed_reference`.

- bias sensitivity was measured on the 3 segment(s) closest to the median duration of 322s, not the longest
- 8 segment(s) were not propagated: no accelerometer channel, or fewer than 300 samples

## 3. Mathematical formulation

Frames are the Phase 3 conventions: navigation ENU (X east, Y north, Z up), vehicle X forward / Y left / Z up, phone as delivered by the logger.

```text
attitude       q_nav_phone(k) = q_nav_phone(k−1) ⊗ δq(ω·Δt)     [exponential map]
specific force f_vehicle      = R_vehicle_phone · f_phone
               f_nav          = R_nav_vehicle · f_vehicle
gravity        a_nav          = f_nav + g_nav                    ← added, not subtracted
velocity       v_k            = v_{k−1} + ½(a_{k−1} + a_k)·Δt
position       p_k            = p_{k−1} + ½(v_{k−1} + v_k)·Δt
```

`a = f + g` is the equation this phase turns on. An accelerometer measures specific force `f = a − g` where `g` points down, so recovering acceleration **adds** the gravity vector. Writing `f − g` leaves a resting device reporting 2g upward — about 19.6 m/s² — which integrates to roughly 300 m of spurious climb in 5.5 s while every intermediate number still looks like a plausible acceleration. Phase 3 hit both halves of this sign question and both were caught by tests rather than by reading.

The attitude increment multiplies on the **right**, because `δθ = ω·Δt` is a rotation of the body. Left-multiplying applies it about navigation axes instead — a different rotation whenever the device is not already level and facing north, and one that produces a perfectly well-formed wrong answer.

Trapezoidal integration is exact for a linearly varying rate, so constant acceleration integrates without error. That is what lets the synthetic tests demand machine precision instead of a tolerance chosen until they passed.

**Not compensated:** coning, sculling, and a midpoint attitude update. Each matters when the rotation rate or specific force varies appreciably *within* one sample interval. At the 10 Hz that dominates this dataset, and against the attitude error measured in §5, they are far from the leading term. The state interface is arranged so they can be added inside `step()` without changing anything a caller sees.

**Numerical safeguards.** Non-finite or implausible sensor samples (specific force beyond 147 m/s², 15 g) are flagged `INVALID_SENSOR` and the state is held rather than sanitized into something plausible. Quaternions are renormalized every step; rotation matrices are validated for orthonormality *and* determinant by the Phase 3 `FrameRotation`.

## 4. Synthetic validation — the only accuracy claim in this phase

Real IO-VNBD has no position ground truth, no attitude ground truth, and a velocity *reference* carrying seconds of residual synchronization error. A real-data run can show the mechanization behaves plausibly; it cannot show it is correct. Synthetic trajectories can, because the answer is known in closed form.

| Case | Max position error (m) |
|---|---|
| constant 1 m/s² acceleration | 4.547e-13 |
| constant 15 m/s velocity | 9.095e-12 |
| irregular sampling | 6.821e-13 |
| pure rotation, gyro propagated | 0.000e+00 |
| stationary (gravity-sign regression) | 0.000e+00 |
| tilted phone, known mounting | 5.087e-12 |

Worst case across every analytical fixture: **9.09e-12 m**. That is float64 rounding, not a tolerance — trapezoidal integration is exact for these profiles, so any real error would stand out by many orders of magnitude.

The fixtures are generated from the definition `f = a − g` written independently of the mechanization's `a = f + g`, so a sign error in one cannot cancel against the other. The stationary case additionally pins the result to a number a reader can check by hand: a level resting device reads +9.80665 m/s² on its up axis, and must compensate to exactly zero.

## 5. Real-data validation

146 smartphone segments were propagated. Selection was on stated criteria — an accelerometer channel and at least 300 samples — never on how the result looked. Segments whose alignment was unavailable are **included**, because their failure behaviour is part of what has to be shown.

| Quantity | Value |
|---|---|
| Segments processed | 146 |
| Propagated without error | 146 |
| With a full phone→vehicle alignment | 94 |
| Alignment unavailable (phone-frame diagnostic) | 52 |
| Compared against a reference | 146 |

### 5.1 Gravity compensation on real data

Across 90 segments containing a detected stationary interval, the mean compensated acceleration during rest has a magnitude of **0.064 m/s²** (median; IQR 0.032–0.112). Its horizontal part has a median of 0.057 m/s² and its vertical part -0.000 m/s².

This is the real-data confirmation of the gravity convention. At rest the true acceleration is zero, so whatever survives compensation is error — and a sign fault would place this distribution at ~19.6 m/s², not near zero.

**The residual is attitude error, and it is the dominant error source.** The *horizontal* component is the one a tilt error produces, because a misjudged vertical tips gravity into east and north. A residual of 0.057 m/s² corresponds to a tilt error of 0.33°. Left uncorrected that integrates to 3.4 m/s of velocity error and 102 m of position error in one minute — and it grows quadratically thereafter. Nothing else in the budget is close: the constant gravity model is worth at most 0.026 m/s², and the missing coning and sculling terms far less at these rates.

The figure worth noting alongside it is that this is the residual at **rest**, where the orientation filter's gravity correction is working under the best conditions it ever sees. During motion the same correction is competing with the vehicle's own acceleration, and the drift rates in §7 imply a substantially larger effective error there. An attitude estimate that is good at rest and degrades under exactly the dynamics you need it for is the central problem Phase 5 inherits.

### 5.2 Speed against the reference

| Comparison | P25 | Median | P75 |
|---|---|---|---|
| Speed RMSE at zero lag (m/s) | 32.60 | 74.31 | 205.88 |
| Speed RMSE at the corrected lag (m/s) | 34.91 | 74.31 | 205.88 |

These are large, and they are supposed to be. An unaided inertial solution on consumer MEMS sensors has no mechanism that would keep the speed bounded: the tilt error in §5.1 is a persistent acceleration error, and a persistent acceleration error is an unbounded velocity error by construction.

**Why the lag-corrected row is not uniformly better.** The lag search maximises *correlation*, which is a shape comparison; the RMSE is dominated by the drift *magnitude*, which a time shift does not touch. On a solution whose speed has run away by tens of m/s, sliding the reference a few seconds changes the residual only marginally and can move it either way. That is not a sign the correction failed — it is the correct reading that on this dataset the residual synchronization error is a small part of the total, and the unaided drift is nearly all of it. Both rows are reported so that split is visible instead of being asserted.

## 6. Reference synchronization methodology

Three different times are kept distinct, because conflating them is how synchronization error gets reported as navigation error:

| Name | What it is |
|---|---|
| Clock alignment | What Phase 2 did — interpolating the vehicle stream onto smartphone sample times. Its contract warns the result is not exact. |
| Residual alignment | The leftover offset Phase 3 measured: median 4.6 s, maximum 14.6 s across 154 segments. Real, large, and corrected nowhere in the stored data. |
| Navigation time | `analysis_time_s` as the mechanization used it. **Never shifted.** |

A lag-corrected comparison therefore resamples **the reference** onto navigation time over a ±15 s search, and reports the shift. The reference at navigation time *t* is the reference recorded at *t + lag*, and it is labelled that way rather than passed off as the raw value at *t*. Shifting the IMU timestamps instead would move the trajectory's own clock to flatter a comparison.

**40** of 146 compared segments showed a distinct correlation peak away from zero, with a median magnitude of 10.2 s. A peak that does not stand clear of the search curve's median is rejected, so a flat curve's argmax is never mistaken for a measurement.

Removing the residual lag changes the median speed RMSE by less than 1% (74.31 → 74.31 m/s). **That is a result, not a null result.** On a dataset where a lag correction mattered, the two would differ substantially, and the gap would be synchronization error misattributed to navigation. Here the unaided drift is so much larger than the timing residual that the split is invisible in the RMSE — which is worth knowing precisely because it will *stop* being true once Phase 5 brings the drift down. The lag must be accounted for then; the measurement is on the comparison object for that purpose.

**Nothing is written back.** The lag is a measurement of leftover Phase 2 error, reported on the comparison object. Canonical timestamps are untouched.

## 7. Drift analysis

A single final error conflates a long drive with a short one, so drift is reported as a rate. Note the units: a constant acceleration error — which is what a tilt error is — produces a *quadratic* position error, so the linear rate below grows with segment duration and is not a constant of the system.

| Quantity | P25 | Median | P75 |
|---|---|---|---|
| Horizontal drift rate (m/s) | 17.63 | 48.78 | 107.54 |
| Final horizontal error (m) | 2,411 | 16,272 | 113,610 |

**A heading caveat that changes how these numbers read.** Where the magnetometer was not trusted, Phase 3's yaw is dead-reckoned from an arbitrary origin, so the propagated east and north components are correct only up to an unknown constant rotation about the vertical. Comparing them directly would measure that unknown rather than the navigation error, so for those segments a constant yaw offset is removed **before** computing position error. The trajectory itself is never modified, the angle removed is reported per segment, and the resulting position error is *blind to a heading error* — it is a lower bound on the true position error, not an estimate of it.

This is not a defect of the mechanization. It is a property of a dataset in which absolute heading is frequently unobservable, and it is precisely the kind of thing a Phase 5 filter with a position update would resolve.

## 8. Bias sensitivity

### 8.1 What a stationary window can actually determine

**Gyroscope: fully.** At rest the true angular rate is zero, so the mean measured rate is the bias. (Earth rate, 7.3e-5 rad/s, is one to two orders of magnitude below a consumer MEMS noise floor and is ignored; a navigation-grade IMU would have to compensate it.)

**Accelerometer: one axis of three.** At rest the sensor reads `f = −g + b`, and averaging cannot separate the two. The direction of gravity in the phone frame is unknown *a priori* — it is exactly what Phase 3 estimates from the same mean — so using that estimate and subtracting yields `b ≡ 0` identically: a number that looks like a measurement and contains none. What *is* separable is the component along gravity, by appealing to the known local gravity magnitude. The two perpendicular components are indistinguishable from the device being tilted slightly differently than believed, and no amount of stationary data resolves them.

So the implementation estimates the gravity-parallel component only, records `observable_axes="gravity_parallel"`, and leaves the other two at exactly zero *by construction rather than by measurement*. Estimates beyond 2 m/s² are rejected as evidence the window was not stationary.

**This is the honest explanation of the drift.** The unobservable horizontal accelerometer bias and the residual tilt error are the same quantity seen two ways, and neither is separable without an independent position or velocity update. That is a filter, and it is Phase 5.

### 8.2 The perturbation experiment

Perturbations are fixed constants drawn from sensor datasheet behaviour (0.001–0.02 rad/s for the gyroscope, 0.005–0.1 m/s² for the accelerometer), applied one phone axis at a time, and **not** tuned against the results. Divergence is measured against the unperturbed run rather than against the reference, which isolates sensitivity to bias from every other error source.

**The two sensors are measured under different attitude modes**, and the reason is worth stating because it is a real property of the baseline rather than a quirk of the experiment. In `PHASE3_FILTER` the attitude comes from the Phase 3 filter, computed upstream from the unperturbed gyroscope; Phase 4 applies its bias to the angular rate *after* that, where nothing consumes it. A gyroscope perturbation therefore moves the baseline trajectory by exactly zero metres — true, and reporting it that way would be badly misleading, because a gyroscope bias matters enormously wherever the gyroscope actually drives attitude. Gyroscope perturbations are therefore measured under `GYRO_PROPAGATION`, against an unperturbed run in that same mode.

| Sensor | Attitude mode | Perturbation | Median divergence (m) |
|---|---|---|---|
| Accelerometer (m/s²) | `phase3_filter` | 0.005 | 258.7 |
| Accelerometer (m/s²) | `phase3_filter` | 0.02 | 1,034.8 |
| Accelerometer (m/s²) | `phase3_filter` | 0.05 | 2,587.0 |
| Accelerometer (m/s²) | `phase3_filter` | 0.1 | 5,174.0 |
| Gyroscope (rad/s) | `gyro_propagation` | 0.001 | 40,286.6 |
| Gyroscope (rad/s) | `gyro_propagation` | 0.005 | 193,762.7 |
| Gyroscope (rad/s) | `gyro_propagation` | 0.01 | 349,282.2 |
| Gyroscope (rad/s) | `gyro_propagation` | 0.02 | 445,012.5 |

Measured over 3 segment(s) of typical length (median 322 s) — the ones closest to the median duration of the aligned set, a rule fixed before any result was seen. Deliberately **not** the longest: a bias error grows quadratically with time, so this dataset's three-hour recordings produce divergences in the hundreds of kilometres, which are arithmetically correct and useless as a description of what a bias costs on an ordinary drive.

For scale, the median path length over these segments is 25,448 m. Read the table against that number rather than in absolute metres: the smallest accelerometer perturbation tried is already a few percent of the distance travelled, and the smallest gyroscope one is several times the distance travelled — because an attitude error grows linearly and drives position error *cubically*, while an accelerometer bias only drives it quadratically.

The point of this table is not the exact metres. It is that perturbations far below what a consumer IMU's run-to-run bias actually is are enough to move the solution by a distance comparable to — for the gyroscope, well beyond — the trajectory itself. **That is the quantitative justification for Phase 5.** An unaided solution cannot be made accurate by better integration; it needs an observation that constrains the bias.

## 9. Performance

Mean **58 µs/sample**, p95 **94 µs/sample** across 146 segments; mean 0.81 s per segment, p95 4.18 s.

These figures include the full Phase 3 chain (orientation filter, gravity, magnetometer assessment, alignment) plus the reference comparison, not the mechanization alone — the mechanization is a small fraction of it. Recorded because the engine must eventually run in real time, **not** because anything here is optimized. Nothing has been tuned for speed and the propagation is a plain Python loop. This is a baseline to improve against.

## 10. Failure cases

**Alignment unavailable.** Phase 3 returns `rotation = None` whenever yaw was never resolved, and Phase 4 never substitutes identity for it — doing so would assert a mounting that was never measured and send the trajectory off in a direction nothing observed. The configured policy decides what happens:

| Policy | Behaviour |
|---|---|
| `REQUIRE` (default) | The run refuses to start, with a message naming the alignment status |
| `PHONE_FRAME_DIAGNOSTIC` | The run proceeds, every sample is flagged `ALIGNMENT_UNAVAILABLE`, and `Trajectory.is_navigation_grade` is False |

The second is legitimate because a strapdown solution actually needs `R_navigation_phone`, which the orientation estimator supplies — the vehicle frame is needed for the vehicle-frame quantities later phases want (longitudinal/lateral split, non-holonomic constraints, wheel-speed comparison), not for the integration itself. The distinction is recorded on the trajectory so a consumer cannot miss it.

52 of 146 segments took this path in the real-data run and are included in every table above.

**Hard failures.** None: every selected segment propagated to completion.

**Invalid time steps.** Steps with Δt ≤ 0 (Phase 2 retains 147,946 duplicate timestamps) or Δt > 0.5 s are not integrated across. The state is held, the sample is flagged `DUPLICATE_TIMESTAMP` or `LARGE_TIME_GAP`, and the trapezoidal accumulator is reset so the next valid step does not average against a rate from before the gap. `SEGMENT` and `FAIL` policies are available for callers who need a break or an exception instead.

## 11. Limitations

1. **No ground truth exists in this dataset.** Every real-data comparison is against a *reference*: Phase 2 rates its best velocity label high-confidence, not truth, and the GNSS position it derives from refreshes about every 9 s. A disagreement is not by itself evidence that the inertial side is the wrong one.
2. **Absolute heading is frequently unobservable.** Where the magnetometer is not trusted, the trajectory is correct only up to a rotation about the vertical. Position errors for those segments have that degree of freedom removed before comparison and are therefore lower bounds.
3. **The horizontal accelerometer bias cannot be estimated here.** It is not separable from a tilt error without an independent position or velocity observation. This is a property of the problem, not a gap in the implementation.
4. **The vertical channel is the least trustworthy.** Gravity compensation error goes straight into it and there is nothing to bound it, which is why horizontal error is reported separately throughout rather than folded into a 3-D norm.
5. **No coning, sculling or midpoint update.** Below the leading error term at these rates, but they will matter once the tilt error is filtered down.
6. **A constant gravity model.** Normal gravity spans 9.780–9.832 m/s² with latitude, so the wrong constant is worth up to 0.026 m/s². `WGS84Gravity` is implemented and substitutable; the baseline does not use it because the term is an order of magnitude below the tilt error.
7. **Stationarity detection is threshold-based**, and its thresholds (0.25 m/s² of magnitude variation, 0.03 rad/s of rotation, 0.15 m/s² from gravity) are engineering constants from sensor physics, not values fitted to this dataset.
8. **The rest detector has a quantified blind spot.** Horizontal acceleration adds to gravity in quadrature, so an acceleration `a` raises the measured magnitude only to √(g² + a²). At a 0.15 m/s² tolerance that leaves accelerations below **1.7 m/s²** indistinguishable from rest — not a defect of this detector but of IMU-only detection, since rest and constant acceleration produce identical specific force. The consequence is bounded here because the result feeds only initialization, bias windows and annotation; it would not be bounded if it fed a zero-velocity correction, which is one reason Phase 4 has none.

## 12. What Phase 5 requires

The baseline says exactly what the next phase has to fix, in priority order set by the measurements above rather than by expectation:

1. **An observation that separates tilt error from horizontal accelerometer bias.** These are the same quantity seen two ways and they dominate everything else. Nothing in an unaided solution can distinguish them; a filter with a position or velocity update can.
2. **A heading observation.** Absolute yaw is unobservable on most segments here, which is why position error can only be reported as a lower bound. Any aiding that constrains heading turns those bounds into measurements.
3. **Accounting for the residual synchronization lag.** Any use of `ref_velocity_mps` as a time-aligned target must handle a multi-second offset — Phase 3 measured a median of 4.6 s correlating accelerometer against dv/dt, and §6 above measures it independently here. `reference_alignment.estimate_residual_lag` reports it per segment, with the acceptance gates that stop it being manufactured; it must not be assumed away.
4. **A like-for-like comparison.** `AttitudeMode`, `AlignmentPolicy`, the gravity model and the bias are all injectable, and the correction budget in §1 is explicit, so a Phase 5 ablation can hold everything else fixed and change one thing.

The interfaces a filter needs exist — `NavigationState` is a complete state, `StrapdownMechanization.step()` is a single propagation step, and the `ImuBias` structure is where an estimated bias would be written. The functionality behind them does not, and deliberately so.

## 13. Figures

- `plots/01_dt_distribution.png`
- `plots/02_resting_residual.png`
- `plots/03_drift_distribution.png`
- `plots/04_error_by_alignment_status.png`
- `plots/05_lag_effect.png`
- `plots/06_bias_sensitivity.png`
- `plots/11_trajectory.png`
- `plots/11_speed.png`
- `plots/11_error.png`
- `plots/11_acceleration.png`
- `plots/12_trajectory.png`
- `plots/12_speed.png`
- `plots/12_error.png`
- `plots/12_acceleration.png`
- `plots/13_trajectory.png`
- `plots/13_speed.png`
- `plots/13_error.png`
- `plots/13_acceleration.png`
- `plots/07_synthetic_recovery.png`

Per-segment figures are drawn for `65348172ad4e:000`, `71453d0af0d3:000`, `fa0b8dc8dddf:000` — chosen to span the outcomes (a well-aligned segment, a weakly aligned one, and one with no alignment at all), not to show the best results.

