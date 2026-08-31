# SIH26168 — Phase 3 Scope Record

Phase 3 built the **sensor-frame, orientation and calibration foundation** that
every later navigation component depends on. It estimates *where the phone is
pointing* and *how the phone relates to the vehicle*. It integrates nothing.

## What Phase 3 delivered

Under `python/idr/frames/`:

- **`conventions.py`** — the frames, axis orders, rotation conventions and
  numerical tolerances, as data rather than as prose scattered through the code.
- **`vectors.py`** — `Vector3` / `VectorSeries` carrying their frame, so a
  triple of floats cannot be silently reinterpreted in the wrong frame.
- **`quaternion.py`** — Hamilton quaternions stored `[x, y, z, w]`, active
  rotations, Shepperd matrix conversion, sign-agnostic comparison.
- **`rotations.py`** — SO(3) validation and `FrameRotation`, which knows which
  two frames it connects and rejects a mismatched application or composition.
- **`gravity.py`** — quasi-static detection and gravity-direction estimation,
  with the specific-force sign convention verified against the real data.
- **`magnetometer.py`** — trust assessment, never correction.
- **`orientation.py`** — a complementary filter over actual Δt, reporting which
  sensor supports which axis.
- **`alignment.py`** — the phone→vehicle engine: tilt from gravity, heading
  from longitudinal dynamics, with confidence and physical checks.
- **`calibration.py`** — the incremental `CalibrationSession` and its states.
- **`synthetic.py`** — controlled rotations and generated drives, the only way
  accuracy can be measured here.
- **`sources.py`**, **`diagnostics.py`**, **`plots.py`**, **`report.py`**,
  **`runner.py`** — canonical-data access and the diagnostic pass.

Outputs: three metadata artifacts in `data/metadata/io_vnbd/`, twelve figures
and the validation report in `reports/sensor_alignment/`, plus
`docs/architecture/sensor_frame_conventions.md` and
`docs/mathematics/phase3_orientation.md`.

## What Phase 3 explicitly did NOT implement

No position or velocity propagation, no strapdown mechanization, no dead
reckoning. No EKF, UKF, filter state or covariance. No GNSS/INS fusion, no
non-holonomic constraints, no map matching. No AI/ML model, training, PyTorch,
or inference. No Android, ONNX, or production C++ navigation. No GNSS outage
simulation.

The interfaces Phase 4 needs exist; the functionality behind them does not.

## Findings that changed the design

Four, each discovered by measurement rather than reasoning:

1. **The specific-force sign.** Both the accelerometer and the gravity channel
   point *away* from gravity at rest — verified across 66 real segments, median
   cosine +0.999. An implementation that negated only one path produced a
   **180° heading error**, caught by a synthetic round trip.
2. **`a = f + g`, not `f − g`.** Recovering linear acceleration *adds* the
   gravity vector. Subtracting it doubles the vertical term, and the result
   still looks like an acceleration.
3. **Residual Phase 2 synchronization error.** The lag between the phone's
   accelerometer and the interpolated reference velocity has a median magnitude
   of ~5 s. At zero lag the forward-direction correlation is ~0.02; corrected,
   it is ~0.73. Phase 3 therefore estimates the lag rather than trusting the
   interpolation — exactly what the Phase 2 contract says to do.
4. **`acos` is the wrong tool for a small angle.** `angular_distance` via
   `2·acos(|a·b|)` put a ~1.5e-8 floor under every comparison. Rewritten with
   `atan2`, the measured round-trip error dropped to 8.7e-16.

## The rule this phase is built around

**Do not fabricate yaw.** Gravity constrains two degrees of freedom; the third
requires motion. A stationary calibration returns roll and pitch with
`rotation = None`, and applying such an estimate raises rather than silently
rotating a caller's data by a placeholder. A confidently wrong 30° heading is
worse than an admitted unknown, because everything downstream inherits it.

## Data safety

Raw data and Phase 2 canonical Parquet are both read-only. A regression test
hashes every input before and after a full run and asserts equality; after the
real-data run, all 564 raw files were re-verified against the Phase 1
acquisition manifest and the canonical Parquet was confirmed unmodified.

## Why this boundary

Rule 10 in
[`../development/engineering_rules.md`](../development/engineering_rules.md):
do not start future phases early. Phase 4 needs a settled answer to "what is
the vehicle-frame acceleration, and how much do we trust it?" — and that answer
turned out to depend on a sign convention, a synchronization residual and a
numerical-precision choice that were all easier to get right in isolation than
inside an INS.
