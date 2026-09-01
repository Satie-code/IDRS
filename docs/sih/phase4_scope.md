# SIH26168 — Phase 4 Scope Record

Phase 4 built the **baseline dead-reckoning engine**: the first code in this
project that integrates anything. It turns Phase 3's orientation and alignment
estimates into a continuously propagated attitude, velocity and position.

It is deliberately **unaided**, and that is the whole point.

## What Phase 4 delivered

Under `python/idr/navigation/`:

- **`state.py`** — `NavigationState`, `InitialState`, the per-sample quality
  flags, and the unit table, declared once rather than assumed everywhere.
- **`gravity.py`** — the gravity *model*, injectable, constant by default.
  Deliberately distinct from Phase 3's gravity *estimator*: one is physics, the
  other is a measurement, and conflating them is how a gravity estimate ends up
  compensated with itself.
- **`bias.py`** — typed gyroscope and accelerometer bias, and an honest account
  of what a stationary window can determine.
- **`integration.py`** — trapezoidal integration on actual Δt, and the
  time-step policy that decides when a step may not be integrated at all.
- **`stationarity.py`** — deterministic rest detection, for initialization and
  annotation. Detection only; there is no zero-velocity correction.
- **`mechanization.py`** — `StrapdownMechanization`: the step, the single
  attitude source, the frame chain, the gravity compensation.
- **`trajectory.py`**, **`geodesy.py`** — the trajectory container and the
  local tangent plane that gives its positions an origin.
- **`reference_alignment.py`** — evaluation only: residual-lag estimation,
  error curves, drift metrics.
- **`synthetic.py`** — analytical trajectories, the only place accuracy can be
  measured in this phase.
- **`diagnostics.py`**, **`plots.py`**, **`report.py`**, **`runner.py`** — the
  real-data pass.

Outputs: three metadata artifacts in `data/metadata/io_vnbd/`, nineteen figures
and the validation report in `reports/navigation/`, plus
`docs/mathematics/phase4_mechanization.md`.

## What Phase 4 explicitly did NOT implement

No EKF, UKF, factor graph or particle filter. No GNSS/INS fusion, no
non-holonomic constraints, no map matching. No zero-velocity *correction* —
detection only. No AI/ML model, training, PyTorch or inference. No Android,
ONNX or production C++. No GNSS-outage simulation.

The interfaces Phase 5 needs exist; the functionality behind them does not.

## The rule this phase is built around

**Do not aid the baseline.** A dead-reckoning baseline is only worth building if
nobody helped it, because its entire purpose is to be the number Phase 5 has to
beat. So the separation is structural rather than a matter of care:

- nothing in `mechanization.py` or anything it calls reads a reference signal;
- `reference_alignment.py`, which does read one, has no path by which it can
  write into a trajectory;
- there is no `update()`, `correct()` or `apply_fix()` method to call, and a
  test asserts their absence;
- the correction budget — what was used, what was not — is written into the
  module docstring and the report, so a later ablation has something concrete
  to hold fixed.

GNSS appears exactly once, as an initial position. The reference velocity
appears exactly once, as an initial speed. After that the trajectory is inertial.

## Findings that changed the design

Six, each discovered by measurement rather than reasoning:

1. **The lag search was measuring drift, not timing.** Correlating the inertial
   speed against the reference speed pinned the search to the edge of its ±15 s
   window on most segments — because an unaided speed carries a large monotone
   drift, and a drifting ramp correlates better and better as the reference is
   shifted. Correlating *rates of change* instead removes the trend and the
   pinning: the lag is now accepted on 40 of 146 segments rather than being
   manufactured on nearly all of them.
2. **The peak-margin gate alone accepts noise.** With 151 candidate lags and
   ~60 independent samples after smoothing, the largest correlation reaches ~0.3
   by chance while the median stays near zero — clearing a 0.1 margin on pure
   noise. An absolute correlation floor of 0.3 closes it.
3. **Stationarity discriminates only in quadrature.** Horizontal acceleration
   raises the measured specific-force magnitude to `√(g² + a²)`, so a 0.5 m/s²
   tolerance was blind to 3.1 m/s² of sustained acceleration — brisk driving,
   classified as rest. Tightened to 0.15 m/s², which leaves a 1.7 m/s² blind
   spot that no IMU-only detector can close, and which is now an asserted test
   rather than a surprise.
4. **The accelerometer bias is two-thirds unobservable at rest.** A horizontal
   bias and a tilt error of `b/g` radians produce identical stationary data.
   Estimating only the gravity-parallel component and marking the other two as
   `observable_axes="gravity_parallel"` is the difference between reporting a
   measurement and reporting a zero that looks like one.
5. **A gyroscope bias is inert in the baseline's attitude mode.** The attitude
   comes from the Phase 3 filter, computed upstream from the unperturbed
   gyroscope; Phase 4's bias is applied afterwards, where nothing consumes it.
   The sensitivity experiment first reported a divergence of exactly zero for
   every gyroscope perturbation — arithmetically correct, and it would have read
   as "gyroscope bias does not matter", which is badly wrong in general.
   Gyroscope perturbations are now measured under `GYRO_PROPAGATION`, and each
   point records the mode that produced it.
6. **The sensitivity experiment must use typical segments, not the longest.**
   A bias error grows quadratically with time, and this dataset contains
   three-hour recordings, so selecting by maximum duration produced divergences
   of hundreds of kilometres — correct arithmetic, useless as a statement of
   what a bias costs on an ordinary drive. Selection is now by closeness to the
   median duration, a rule fixed before any result is read.

## What the baseline actually does

It drifts, quickly, and the report says so in numbers rather than adjectives.
The dominant term is attitude error leaking gravity into the horizontal
channels — the same quantity as the unobservable horizontal accelerometer bias,
seen from the other side. Neither is separable without an independent position
or velocity observation, which is a filter, which is Phase 5.

That is not a defect to be tuned away. An unaided inertial solution on consumer
MEMS sensors has no mechanism that would bound its velocity error, and a
baseline that appeared not to drift would mean something had quietly helped it.

## Honest limits on the real-data numbers

- IO-VNBD documents **no ground truth** for position, velocity or attitude.
  Every real comparison is against a *reference* with known defects.
- Where the magnetometer is untrusted, absolute heading is unobservable, so the
  reported position errors have one rotational degree of freedom removed and are
  **lower bounds**, not estimates.
- The residual Phase 2 synchronization error is real and is measured per
  segment, never written back.

## Data safety

Raw data and Phase 2 canonical Parquet are both read-only. A regression test
hashes every input before and after a full run and asserts equality, and a
second asserts that no new file appears in the canonical tree. Canonical
timestamps are never shifted — the lag correction moves the *reference*.

## Why this boundary

Rule 10 in
[`../development/engineering_rules.md`](../development/engineering_rules.md):
do not start future phases early. Phase 5 needs a settled, trustworthy answer to
"what does the physics alone give us, and how fast does it fall apart?" — and
that answer turned out to depend on a sign convention, an observability
argument, a correlation methodology and a detection threshold that were all far
easier to get right in isolation than inside a filter.
