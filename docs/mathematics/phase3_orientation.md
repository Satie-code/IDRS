# Phase 3 Mathematics — Orientation and Alignment

The equations actually implemented in
[`python/idr/frames/`](../../python/idr/frames/). Conventions are fixed in
[`../architecture/sensor_frame_conventions.md`](../architecture/sensor_frame_conventions.md);
this document assumes them.

Notation: `q_A_B` is the unit quaternion whose rotation matrix `R_A_B` maps
B-frame coordinates to A-frame coordinates. Vectors are column vectors.

---

## 1. Quaternion algebra

A unit quaternion is stored `q = [x, y, z, w]` with vector part
`v = (x, y, z)` and scalar `w`, representing a rotation of angle `θ` about unit
axis `n`:

```text
q = [ n·sin(θ/2),  cos(θ/2) ]
```

### Hamilton product

For `a = [a_v, a_w]` and `b = [b_v, b_w]`:

```text
a ⊗ b = [ a_w·b_v + b_w·a_v + a_v × b_v ,   a_w·b_w − a_v · b_v ]
```

Written out in the storage order the code uses:

```text
(a ⊗ b)_x = a_w b_x + a_x b_w + a_y b_z − a_z b_y
(a ⊗ b)_y = a_w b_y − a_x b_z + a_y b_w + a_z b_x
(a ⊗ b)_z = a_w b_z + a_x b_y − a_y b_x + a_z b_w
(a ⊗ b)_w = a_w b_w − a_x b_x − a_y b_y − a_z b_z
```

This corresponds to matrix multiplication **in the same order**:

```text
R(a ⊗ b) = R(a) · R(b)
```

which is what makes `q_A_C = q_A_B ⊗ q_B_C` valid. A test asserts the identity
against explicit matrix products rather than trusting the derivation.

### Vector rotation

Implemented in the cross-product form, avoiding an intermediate matrix:

```text
t   = 2 · (v × u)
u'  = u + w·t + v × t
```

for `q = [v, w]` and vector `u`. Fewer operations than building `R` for a
single vector, and one less rounding stage. For an `(n, 3)` block the matrix
form wins instead, and that is what `rotate_vectors` uses.

### Inverse and conjugate

```text
q*    = [ −v,  w ]
q⁻¹   = q* / ‖q‖²
```

For a unit quaternion the two coincide. `inverse` uses the full form so it
remains correct for a non-unit input rather than silently returning something
that is not an inverse.

### Rotation matrix

```text
        ⎡ 1−2(y²+z²)   2(xy−wz)    2(xz+wy) ⎤
R(q) =  ⎢ 2(xy+wz)    1−2(x²+z²)   2(yz−wx) ⎥
        ⎣ 2(xz−wy)     2(yz+wx)   1−2(x²+y²)⎦
```

### Matrix → quaternion (Shepperd's method)

The naive route takes `w = √(1 + tr R)/2` and divides the off-diagonal
differences by `4w`. That divisor vanishes as the rotation approaches 180°,
losing most of the available precision exactly where rotations are still
perfectly ordinary.

Instead, the largest of the four components is identified from the trace and
the diagonal, and the other three are derived from it. The divisor is then
bounded away from zero in every branch. Measured worst-case round-trip error
over 200 random rotations: **8.7 × 10⁻¹⁶ rad**.

### Angular distance

```text
d(a, b) = 2 · atan2( ‖vec(a⁻¹ ⊗ b)‖ , |w(a⁻¹ ⊗ b)| )
```

Mathematically equal to `2·acos(|a · b|)`, but numerically very different.
`acos` has an infinite derivative at 1: two rotations one ulp apart measure as
~1e-8 rad apart under `acos` and ~1e-16 under `atan2`. Taking the absolute
value of the scalar part makes the result sign-agnostic, so `q` and `−q` are
correctly reported as identical, and the range is `[0, π]`.

### Euler angles (Z-Y-X intrinsic) — diagnostics only

```text
R = Rz(ψ) · Ry(θ) · Rx(φ)
```

Forward:

```text
x = sin(φ/2)cos(θ/2)cos(ψ/2) − cos(φ/2)sin(θ/2)sin(ψ/2)
y = cos(φ/2)sin(θ/2)cos(ψ/2) + sin(φ/2)cos(θ/2)sin(ψ/2)
z = cos(φ/2)cos(θ/2)sin(ψ/2) − sin(φ/2)sin(θ/2)cos(ψ/2)
w = cos(φ/2)cos(θ/2)cos(ψ/2) + sin(φ/2)sin(θ/2)sin(ψ/2)
```

Inverse:

```text
sin θ = 2(wy − zx)                     (clamped to [−1, 1])
φ     = atan2( 2(wx + yz), 1 − 2(x² + y²) )
ψ     = atan2( 2(wz + xy), 1 − 2(y² + z²) )
```

**At |θ| = 90° the decomposition is degenerate.** Substituting
`sin(θ/2) = ±cos(θ/2) = ±√2/2` into the forward equations gives

```text
2·atan2(x, w) = φ − ψ    at θ = +90°
2·atan2(x, w) = φ + ψ    at θ = −90°
```

so only `ψ − φ` (or `ψ + φ`) is determined. The implementation sets `φ = 0` and
assigns the invariant to `ψ`, negating in the `+90°` case. Getting that sign
backwards returns `φ − ψ` as the yaw and rebuilds a *different* rotation — a
defect the round-trip test caught. `euler_is_well_conditioned(q)` reports
whether the split is meaningful at all.

---

## 2. Gravity estimation

An accelerometer measures specific force:

```text
f = a − g
```

with `g` the gravity vector (pointing down, ‖g‖ ≈ 9.80665 m/s²). At rest
`f = −g`, so **the sensor points up** and both the accelerometer and the
logger's gravity channel must be negated to yield `ĝ`.

### Quasi-static detection

A sample qualifies when both hold:

```text
| ‖f‖ − g |  ≤  0.5 m/s²
‖ω‖          ≤  0.1 rad/s
```

The second condition is not optional. A vehicle in a steady turn holds `‖f‖`
near 1 g while the direction sweeps, so the magnitude test alone accepts it and
the average smears across the sweep.

**Known blind spot.** For acceleration `a` perpendicular to gravity,

```text
‖f‖ = √(g² + a²)   ⟹   ‖f‖ − g ≈ a²/(2g)
```

so `a = 2 m/s²` shifts the magnitude by only 0.2 m/s² and passes a 0.5 m/s²
gate. Such a sample tilts the estimated vertical by up to `atan(a/g)`. Over a
window with mixed directions the errors largely cancel; over a consistently
accelerating window they do not. The reported direction spread and confidence
are what surface it.

### Estimate and confidence

```text
ĝ = − mean(f_i) / ‖mean(f_i)‖         over qualifying samples
```

Confidence is a product of three named factors, so a low score is always
attributable to one of them:

```text
consistency = max(0, 1 − spread / 0.35 rad)
magnitude   = max(0, 1 − |mean‖f‖ − g| / (0.2 g))
support     = 0.5 + 0.5 · min(1, n / (10 · n_min))
confidence  = consistency · magnitude · support
```

where `spread` is the RMS angle between each sample direction and the mean.
Magnitude is judged on the mean of the norms, not the norm of the mean:
averaging vectors that point different ways shortens the result, and that
shortening is the spread's business.

### Linear acceleration

```text
a = f + g
```

The gravity vector is **added**. Subtracting it doubles the vertical term
rather than cancelling it, and the result still looks like an acceleration.

---

## 3. Orientation filter

A complementary filter producing `q_navigation_phone`. Not a Kalman filter: a
state vector with a covariance is Phase 4/5 work.

### Propagation

Over a measured interval `Δt` with body angular rate `ω`:

```text
q ← q ⊗ exp(ω Δt)     where exp(ω Δt) = [ (ω/‖ω‖)·sin(‖ω‖Δt/2),  cos(‖ω‖Δt/2) ]
```

The exact axis-angle exponential, not the first-order `q + ½Ω q Δt`. The exact
form stays on the unit sphere by construction and costs one sin/cos; the
first-order form must be renormalized every step and still accumulates error
proportional to the rotation rate — precisely when accuracy matters.

`Δt` comes from `analysis_time_s` sample by sample. Steps are **skipped**, not
bridged, when `Δt ≤ 0` (a duplicate timestamp, which Phase 2 retains and flags)
or `Δt > 0.5 s`, and when `‖ω‖ > 35 rad/s`, which is beyond any consumer MEMS
gyro's range and therefore saturation rather than motion.

### Gravity correction

Applied only when `| ‖f‖ − g | ≤ 0.5 m/s²`, so the filter is never pulled
toward an accelerometer that is currently measuring braking:

```text
d_meas = −R(q) · f/‖f‖              measured down, in navigation coordinates
d_exp  = (0, 0, −1)
Δq     = shortest rotation d_meas → d_exp
q      ← axis_angle(axis(Δq), angle(Δq) · min(1, k_g·Δt)) ⊗ q
```

with `k_g = 0.5 Hz`, giving a correction time constant of about two seconds:
fast enough to bound gyro drift over a long drive, slow enough that a few
seconds of hard braking cannot tip the estimate appreciably.

### Magnetic correction

Applied only when the magnetometer passes its quality checks, and constrained
to the **vertical axis**:

```text
b_nav  = R(q) · m/‖m‖
ε      = atan2( b_nav,x , b_nav,y )        heading error from north (ENU: +Y)
q      ← axis_angle( (0,0,1), −ε · min(1, k_m·w·Δt) ) ⊗ q
```

with `k_m = 0.05 Hz` and `w` the magnetometer's trust weight. Two deliberate
restrictions:

- **Only the horizontal field component is used.** A near-vertical field in the
  navigation frame carries no heading and is skipped.
- **Only the vertical axis is corrected.** Letting the magnetometer influence
  roll and pitch would let a disturbed field undo the gravity correction, which
  is the better-founded of the two.

`k_m` is an order of magnitude below `k_g` because the magnetometer's
in-vehicle failure mode is a *steady* bias, and a strong gain would track that
bias faithfully into the estimate.

### Axis support

Reported per segment rather than implied:

| Axis | `OBSERVED` when | Otherwise |
|---|---|---|
| roll, pitch | gravity is available | `UNRESOLVED` |
| yaw | the magnetometer is trusted | `DEAD_RECKONED` from the gyro, or `UNRESOLVED` with no gyro |

Confidence is capped at 0.5 whenever yaw lacks an absolute reference.

---

## 4. Magnetometer quality

No correction, no calibration — only a trust weight, with the reasons kept
separate. Four independent checks:

```text
magnitude    ‖m‖ ∈ [25, 65] µT for ≥80% of samples
stability    std(‖m‖)/mean(‖m‖) ≤ 0.15
slew         d(direction)/dt ≤ 3 rad/s
dip          p95 − p05 of angle(m, ĝ) ≤ 15°
```

A rigid rotation in a fixed field does not change `‖m‖`, so instability is
direct evidence the field is not fixed. Dip cannot be checked against an
absolute value — it varies from 0° at the magnetic equator to ±90° at the poles
— but within one short window it should barely move.

Two independent findings drop the weight to zero. A median magnitude beyond
twice the plausible maximum drops it on that ground alone: that is not the
Earth's field with noise on it.

---

## 5. Phone → vehicle alignment

Two halves with very different difficulty.

### Tilt — 2 DOF, from gravity

Vehicle up in phone coordinates is `û = −ĝ`. This fixes two degrees of freedom
and is available while stationary.

### Heading — 1 DOF, requires motion

Rotating the phone about the vertical leaves every gravity reading identical,
so gravity carries **no** heading information. Yaw needs motion.

Under longitudinal acceleration the vehicle's specific force in the horizontal
plane lies along ±forward. With `h_i` the horizontal linear acceleration and
`s_i = dv/dt` from an independent scalar speed reference, the least-squares
forward direction is

```text
f ∝ Σ h_i s_i,        h_i = a_i − (a_i · û) û
```

re-projected onto the horizontal plane afterwards. The **sign** of the
covariance separates forward from reverse, which a magnitude-only method cannot.

Both signals are first low-passed with a trailing 2 s time-window average. A
vehicle's longitudinal acceleration is a sub-Hz quantity; a phone's
accelerometer is dominated by road vibration. On real IO-VNBD segments the
horizontal specific force has a median magnitude of ≈1.2 m/s² against a vehicle
`dv/dt` median of ≈0.37 m/s², and correlating them unfiltered yields ≈0.02.

### Lag estimation

Phase 2 measured a per-pair smartphone/vehicle offset and deliberately left
both streams as recorded, and its contract warns against assuming the
synchronization is exact. Measured across 16 real segments, the residual lag
between the phone's accelerometer and the interpolated reference velocity has a
median magnitude of **5.2 s**; correcting it lifts the median correlation from
0.02 to 0.73.

So the lag is estimated rather than assumed:

```text
lag* = argmax_lag  corr( h · f(lag),  s(t + lag) )      over ±15 s
```

with `s` resampled by interpolation on the actual timestamps — not `np.roll`,
which would assume a uniform rate this dataset does not have. The peak must
exceed the median of the search curve by 0.1, otherwise a flat curve's argmax
would be mistaken for a measurement and no lag is applied.

**Nothing is written back.** The lag is reported on
`AlignmentQuality.forward_lag_s`; the canonical Parquet is untouched.

### Assembling the rotation

```text
û  = −ĝ                              exact, from gravity
f̂  = normalize( f − (f · û) û )       azimuth only
l̂  = û × f̂
R_phone_vehicle = [ f̂  l̂  û ]         columns
R_vehicle_phone = R_phone_vehicleᵀ
```

`û` is taken as exact and `f̂` supplies only the azimuth. Averaging a noisy
heading into the vertical would trade a well-measured axis for a poorly-measured
one. If `f̂` is parallel to `û` it carries no azimuth and the construction
raises rather than returning an arbitrary basis.

### Acceptance

An alignment is `SUCCESS` only when gravity is confident, the forward
correlation is at least 0.3, disjoint halves of the data agree within 15°, and
the transformed gravity lands within 10° of vehicle vertical. Otherwise the
status names what was missing, `rotation` is `None`, and applying the estimate
raises. **Yaw is never fabricated.**

### Physical checks

Applied to the candidate alignment. Diagnostics, not proofs:

```text
gravity        angle( R·ĝ , (0,0,−1) )  ≤ 10°
longitudinal   corr( a_vehicle,x , dv/dt )        should be strongly positive
lateral        corr( a_vehicle,y , v·ω_z )        centripetal: left turn → +Y
angular        corr( ω_vehicle,z , reference yaw rate )
```

The gravity check alone is insufficient: a pure yaw error leaves it perfectly
satisfied while putting forward acceleration on the lateral axis. That is why
the longitudinal check exists, and a test asserts exactly that failure mode.

---

## 6. What is deliberately absent

No position or velocity propagation, no strapdown integration, no filter state
or covariance, no GNSS fusion, no non-holonomic constraint, no map matching, no
learned component. Phase 3 produces an orientation, an alignment, and the
quality metrics behind them. Phase 4 consumes them.
