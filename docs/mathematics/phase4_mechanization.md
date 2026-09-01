# Phase 4 — Strapdown Inertial Mechanization

The equations the baseline dead-reckoning engine implements, the conventions
they depend on, and the terms deliberately left out. The machine-readable
version is [`python/idr/navigation/`](../../python/idr/navigation/); the frames
and rotation conventions come from
[`../architecture/sensor_frame_conventions.md`](../architecture/sensor_frame_conventions.md)
and are not restated here except where a sign matters.

Measured results are in
[`../../reports/navigation/Phase4_Inertial_Mechanization_Validation.md`](../../reports/navigation/Phase4_Inertial_Mechanization_Validation.md).

---

## 1. Frames and notation

| Frame | Axes | Symbol |
|---|---|---|
| Navigation | X east, Y north, Z up (ENU) | `n` |
| Vehicle | X forward, Y left, Z up | `v` |
| Phone | as delivered by the logger; physical directions unverified | `p` |

A rotation is named destination-first, so `R_n_p` maps phone coordinates into
navigation coordinates:

```text
x_n = R_n_p · x_p
```

Quaternions are Hamilton, stored `[x, y, z, w]`, and represent active rotations.
`q_n_p` is the quaternion form of `R_n_p`.

Units are fixed for the whole package: metres, m/s, m/s², rad/s, seconds.
Degrees appear only in diagnostics.

---

## 2. Specific force — the definition everything rests on

An accelerometer does not measure acceleration. It measures **specific force**:

```text
f = a − g
```

where `a` is the acceleration of the sensor in an inertial frame and `g` is the
gravitational acceleration vector, **pointing down**. At rest `a = 0`, so

```text
f = −g          |f| = 9.80665 m/s²,  pointing up
```

A phone lying flat and screen-up reads approximately `(0, 0, +9.81)`. That is a
number a reader can check by hand, and every sign in this phase is anchored to
it.

Rearranging gives the equation the mechanization uses:

```text
a = f + g                                                            (2.1)
```

**The gravity vector is added, not subtracted.** Writing `f − g` leaves a
resting device reporting `2g` upward, which integrates to about 300 m of
spurious climb in 5.5 s — while every intermediate quantity still looks like a
plausible acceleration, so nothing downstream complains. Phase 3 shipped both
halves of this sign question at different moments: the direction of the gravity
*estimate* (which produced a 180° heading error) and the sign of this *sum*.
Both were caught by tests rather than by reading, which is why
`test_gravity_sign_regression` asserts the stationary case explicitly.

Phase 3 additionally verified against the real dataset that the accelerometer
channel and the logger's gravity channel agree in sign — median cosine +0.999
across 66 segments with a quasi-static interval — so both read specific force
and both must be negated to obtain `g`.

---

## 3. Gravity model

The baseline is constant:

```text
g_n = [0, 0, −g₀]ᵀ,     g₀ = 9.80665 m/s²                            (3.1)
```

the same standard gravity Phase 2 used for its g→m/s² unit conversion and
Phase 3 used for its magnitude checks. One value across four phases means a
vehicle acceleration expressed in g and a gravity compensation cannot disagree
by a rounding choice.

The model is injectable (`GravityModel`). `WGS84Gravity` implements Somigliana
normal gravity,

```text
γ(φ) = γ_e · (1 + k·sin²φ) / √(1 − e²·sin²φ)                         (3.2)
γ(φ, h) = γ(φ) − 3.086e−6 · h
```

with `γ_e = 9.7803253359 m/s²`, `k = 0.00193185265241`,
`e² = 0.00669437999013`. It is **not** used by the baseline. Normal gravity
spans 9.780 m/s² at the equator to 9.832 m/s² at the poles, so the wrong
constant costs at most ~0.026 m/s² — real, but an order of magnitude below the
attitude error measured in the validation report, and therefore not the term to
fix first.

---

## 4. Attitude propagation

Over an interval `Δt` with body-frame angular rate `ω`, the rotation the body
undergoes is `δθ = ω·Δt`. Because it is a rotation *of the body*, the increment
multiplies on the **right**:

```text
q_n_p(k) = q_n_p(k−1) ⊗ δq(δθ)                                       (4.1)

δq(δθ) = [ (δθ/‖δθ‖)·sin(‖δθ‖/2),  cos(‖δθ‖/2) ]                     (4.2)
```

Left-multiplying instead applies the increment about *navigation* axes, which
is a different rotation whenever the device is not already level and facing
north — and it yields a perfectly well-formed unit quaternion, so nothing but a
test will catch it.

Equation (4.2) is the exponential map and is **exact for a constant rate over
the interval**, not a small-angle approximation: a 90°-per-step rotation
propagates correctly. Below `‖δθ‖ = 1e−12` the axis is undefined and the
increment is taken as identity rather than normalizing a near-zero vector into
a confident direction. The quaternion is renormalized every step; over 10⁴
steps the norm holds to within 1e−12.

`Δt` is always `t_k − t_{k−1}` from `analysis_time_s`. No nominal rate is
assumed anywhere — Phase 2 measured families at roughly 2, 10, 24 and 1000 Hz in
this dataset.

### One attitude source, never two

Phase 4 offers two modes and uses exactly one per run:

| Mode | Attitude |
|---|---|
| `PHASE3_FILTER` (baseline) | The Phase 3 complementary filter's output at each sample. That filter already fuses gyroscope, gravity and — where trusted — the magnetometer, on actual Δt. |
| `GYRO_PROPAGATION` | Equation (4.1) alone from the initial attitude, never corrected. |

Taking the Phase 3 estimate *and* integrating the gyroscope *and* correcting
with gravity would count the same measurements two or three times, and the
result would look better than either input while being answerable to neither.
The separation is structural: in `PHASE3_FILTER` the propagator is never
called, and in `GYRO_PROPAGATION` a supplied orientation is ignored.

**A consequence worth stating, because it surprised this implementation.** In
`PHASE3_FILTER` the attitude is produced upstream, from the *unperturbed*
gyroscope; the Phase 4 gyroscope bias of §9 is subtracted from the angular rate
afterwards, where nothing consumes it. The baseline is therefore **exactly
insensitive to a gyroscope bias** — a perturbation of any size moves the
trajectory by zero metres. That is a true statement about this configuration and
a badly misleading one about inertial navigation, so the bias-sensitivity
experiment measures gyroscope perturbations under `GYRO_PROPAGATION` instead,
against an unperturbed run in the same mode, and labels every result with the
mode that produced it.

---

## 5. Frame transformation

```text
f_v = R_v_p · f_p                                                    (5.1)
R_n_v = R_n_p · R_v_pᵀ                                               (5.2)
f_n = R_n_v · f_v                                                    (5.3)
```

`R_v_p` is the Phase 3 `AlignmentEstimate.rotation` — the mounting rotation,
estimated from data, never assumed. `R_n_p` is the attitude from §4.

Equations (5.1)–(5.3) compose to `f_n = R_n_p · f_p`, so routing through the
vehicle frame is algebraically free and gives `f_v` — the longitudinal/lateral
split every later phase needs — as a by-product.

**When `R_v_p` is unavailable**, Phase 3 returns `rotation = None` because yaw
was never resolved. Phase 4 never substitutes identity for it. The configured
policy either refuses to run (`REQUIRE`, the default) or proceeds without the
vehicle frame (`PHONE_FRAME_DIAGNOSTIC`), flagging every sample and marking the
trajectory not navigation-grade. The second is legitimate because a strapdown
solution needs `R_n_p`, not the vehicle frame; what it loses is every
vehicle-frame quantity, not the integration.

---

## 6. Gravity compensation

```text
a_n = f_n + g_n                                                      (6.1)
```

Equation (2.1) in the navigation frame. This is the single line the phase turns
on, and §11 of the validation report quantifies what survives it on real data:
during detected rest, where the true `a_n` is zero, the residual is the error.

---

## 7. Velocity and position integration

Trapezoidal, on actual timestamps:

```text
v_k = v_{k−1} + ½·(a_{k−1} + a_k)·Δt                                 (7.1)
p_k = p_{k−1} + ½·(v_{k−1} + v_k)·Δt                                 (7.2)
```

Trapezoidal rather than rectangular because the rectangular rule's error is
first order in `Δt` **and biased in one direction** — it under-integrates a
rising signal at every step, so a vehicle accelerating from rest accumulates a
systematic velocity deficit rather than a zero-mean error. Trapezoidal is
second order and exact for a linearly varying rate, which covers constant
acceleration exactly. That is what allows the synthetic tests to demand
agreement at machine precision (measured worst case ~1e−13 m over 60 s) rather
than at a tolerance chosen until they passed.

The first sample is not integrated: it carries the initial state verbatim, so
`v = v₀ + at` and `p = p₀ + v₀t + ½at²` hold from index 0 and an off-by-one in
the first step cannot hide inside a tolerance.

Note (7.2) uses `v_k`, computed by (7.1) in the same step. The two together are
equivalent to a second-order position update and are not an implicit scheme —
`v_k` depends only on quantities already known.

---

## 8. Time-step policy

`Δt` is classified before every step:

| Verdict | Condition | Handling |
|---|---|---|
| `VALID` | `0 < Δt ≤ 0.5 s` | Integrated |
| `NON_POSITIVE` | `Δt ≤ 0` | Not integrated; flagged `DUPLICATE_TIMESTAMP` |
| `TOO_LARGE` | `Δt > 0.5 s` | Not integrated; flagged `LARGE_TIME_GAP` |
| `NOT_FINITE` | NaN or ∞ | Not integrated; flagged `INVALID_SENSOR` |

The 0.5 s limit matches `idr.frames.orientation.MAX_INTEGRATION_STEP_S`, so
attitude and velocity break at the same places — a trajectory whose attitude was
held across a gap its velocity integrated through would be internally
inconsistent in a way nothing downstream could detect. Across a gap longer than
this an inertial solution has no information about the interior, and
trapezoidal integration between the endpoints is an assumption dressed as a
measurement.

Three configurable actions: `SKIP` (hold the state, flag, continue — the
default), `SEGMENT` (end the trajectory, require re-initialization), `FAIL`
(raise). After any held step the trapezoidal accumulator is reset, so the next
valid step does not average against a rate from before the gap.

A step is additionally flagged `IRREGULAR_DT` when it departs from the
*segment's own median* by more than 5×. Relative, not absolute: 0.4 s is
ordinary at 2 Hz and a forty-fold outlier at 1000 Hz. Irregular steps are still
integrated — irregular sampling is a property of the logger, not an error.

Duplicate timestamps are never dropped from the source. Phase 2 retains 147,946
of them by design; Phase 4 declines to integrate across them, counts them, and
leaves the canonical data untouched.

---

## 9. Bias

The sensor model is additive in the phone frame:

```text
ω_measured = ω_true + b_g                                            (9.1)
f_measured = f_true + b_a                                            (9.2)
```

so correction subtracts, and must do so **before** any frame transformation — a
bias applied after a rotation subtracts a constant from the wrong axes and the
result stays plausible.

Phase 4 accepts a fixed bias from configuration or estimates one once from a
stationary window. It does not estimate bias adaptively; that is a filter state.

### What a stationary window can determine

**Gyroscope: fully.** At rest `ω_true = 0`, so the window mean *is* `b_g`.
Earth rate (7.292115e−5 rad/s) is ignored: it is one to two orders of magnitude
below a consumer MEMS noise floor. A navigation-grade IMU would have to
compensate it.

**Accelerometer: one axis of three.** At rest the sensor reads

```text
f_measured = −g_p + b_a                                              (9.3)
```

and averaging cannot separate the two terms:

- The direction of `g_p` is unknown *a priori* — it is exactly what Phase 3
  estimates from this same mean. Using that estimate in (9.3) and subtracting
  gives `b_a ≡ 0` identically: a number that looks like a measurement and
  contains none.
- What *is* separable is the component along gravity, by appealing to an
  external fact — the local gravity magnitude:

  ```text
  b_∥ = (f̄ · û) − g₀,     û = −ĝ_p                                   (9.4)
  ```

- The two components perpendicular to `û` remain unobservable. A horizontal
  bias `b_⊥` and a tilt of `b_⊥/g₀` radians produce **identical** stationary
  data. No amount of stationary averaging separates them.

The implementation therefore returns `b_a = b_∥·û` and records
`observable_axes = "gravity_parallel"`, so the two zeros are not mistaken for
measurements.

This is the honest explanation of the drift: the unobservable horizontal
accelerometer bias and the residual tilt error are the same quantity seen two
ways. Separating them requires an independent position or velocity observation —
that is a filter, and it is Phase 5.

---

## 10. Stationarity detection

Three conditions over a trailing 2 s window, all required:

```text
σ(|f|)  ≤ 0.25 m/s²        no translational dynamics
mean|ω| ≤ 0.03 rad/s       no rotation
| mean|f| − g₀ | ≤ 0.15    the magnitude a resting sensor reads
```

The third earns its place: a vehicle in smooth cruise passes the first two
comfortably.

**Its blind spot, stated with a number.** Horizontal acceleration adds to
gravity *in quadrature*: an acceleration `a` raises the measured magnitude only
to `√(g₀² + a²)`, so a tolerance `τ` leaves the detector blind to accelerations
below `√(2·g₀·τ)`. At `τ = 0.15` that is **1.7 m/s²**. A vehicle holding a
gentle, perfectly constant acceleration in a straight line is therefore
classified as stationary — not by this detector specifically, but by any
IMU-only detector, because rest and constant acceleration are genuinely
indistinguishable from specific force alone.

The consequence is bounded here because the result is used only for
initialization, bias windows and annotation: a mis-detected window yields a bias
estimate the plausibility check rejects, or an initial velocity of zero recorded
as an assumption. It would **not** be bounded if this fed a zero-velocity
correction, which is one reason Phase 4 has none.

`τ = 0.15` is not arbitrary. Across the real segments where a stationary
interval was found, the mean specific-force magnitude during rest sits between
9.80 and 9.90 m/s² — a worst-case departure of 0.09.

---

## 11. Initial state

```text
NavigationState = (t, p_n, v_n, q_n_p)
```

**Position.** The first fresh GNSS fix, which is also the tangent-plane origin,
so the trajectory starts at `(0, 0, 0)` by construction. GNSS is read here and
nowhere else: it is an initialization value, not a continuous input, and nothing
in the mechanization can reach it.

**Velocity.** Zero when the segment opens with a detected stationary interval —
the defensible case. Otherwise the reference velocity read at the
**lag-corrected** instant (§12), because picking the row whose timestamp matches
would import a median 4.6 s of residual synchronization error straight into the
initial state. Its direction comes from the initial attitude's forward axis,
which is the only self-consistent choice; using the GNSS course would import a
second reference. When neither is available, `v₀ = 0` is used and recorded as an
*assumption*, and every velocity downstream inherits that error.

**Attitude.** The Phase 3 orientation at the first sample. When the
magnetometer was not trusted its yaw is an arbitrary origin rather than a
measurement, and `heading_observed = False` records that. See §13.

### The tangent plane

Latitude and longitude are projected to local ENU about a stated origin, using
the WGS-84 radii of curvature evaluated there:

```text
north = (φ − φ₀) · (M(φ₀) + h₀)
east  = (λ − λ₀) · (N(φ₀) + h₀) · cos φ₀                            (11.1)

M(φ) = a(1 − e²) / (1 − e²sin²φ)^{3/2}      N(φ) = a / √(1 − e²sin²φ)
```

Exact to first order in the displacement; the second-order error over a 10 km
excursion is under a metre. It is not a general map projection and must not be
used as one. The origin is the **first** valid fix, not the centroid: a
propagated position starts at zero, and a reference centred on the middle of the
drive would begin hundreds of metres away, producing a constant offset
indistinguishable from an initialization error.

---

## 12. Reference comparison — three different times

Evaluation only. Nothing in this section can write into a trajectory.

| Name | What it is |
|---|---|
| Clock alignment | What Phase 2 did: interpolate the vehicle stream onto smartphone sample times. Its contract warns the result is not exact. |
| Residual alignment | The leftover offset Phase 3 measured — median 4.6 s, maximum 14.6 s over 154 segments. Corrected nowhere in the stored data. |
| Navigation time | `analysis_time_s` as the mechanization used it. **Never shifted.** |

A lag-corrected comparison resamples **the reference** onto navigation time:

```text
r̃(t) = r(t + λ)                                                     (12.1)
```

and reports `λ`. The reference at navigation time `t` is the reference recorded
at `t + λ`, labelled as such. Shifting the IMU timestamps instead would move the
trajectory's own clock to flatter a comparison.

### Estimating λ

The search correlates **rates of change**, not speeds:

```text
λ̂ = argmax_λ  corr( S[d(|v_ins|)/dt],  S[d(r̃_λ)/dt] )               (12.2)
```

where `S[·]` is a 2 s trailing mean. Differencing is not cosmetic. An unaided
inertial speed carries a large near-monotone drift, and correlating a drifting
ramp against anything monotone scores high everywhere and *improves* as the
reference is shifted — so a raw-speed search saturates at the edge of its window
and reports a confident lag that measures the drift rather than the timing.
Differencing removes a linear trend exactly and a slow drift very nearly. The
2 s smoothing matches Phase 3's: differentiating a 10 Hz speed amplifies exactly
the road-vibration band that carries no vehicle dynamics.

A peak is accepted only if it clears **both** gates:

```text
corr(λ̂) − median_λ corr(λ)  ≥  0.10        the peak is a peak
corr(λ̂)                     ≥  0.30        it is a real correlation
```

The second gate exists because of multiple comparisons. The search evaluates 151
candidates, and after 2 s smoothing the signals carry roughly one independent
sample every 2 s — about 60 in a two-minute segment. The largest of 151 draws
with standard error 1/√60 ≈ 0.13 reaches ~0.3 by chance while the median stays
near zero, which clears the margin gate on **pure noise**. The absolute floor
closes that; 0.30 matches Phase 3's `MIN_FORWARD_CORRELATION`.

Both the zero-lag and the lag-corrected comparison are always produced. The
difference between them is the amount of apparent navigation error that is
really synchronization error, and reporting only one would misattribute it.

---

## 13. The heading ambiguity in position error

Where the magnetometer is not trusted, Phase 3's yaw is dead-reckoned from an
arbitrary origin. The propagated east and north components are then correct only
up to an unknown constant rotation about the vertical, and comparing them
directly to a geo-referenced track measures that unknown rather than the
navigation error.

For those segments — and only those — a constant yaw offset is removed before
computing position error. It is a closed form, not a search:

```text
θ̂ = atan2( Σ(pₓ·rᵧ − pᵧ·rₓ),  Σ(pₓ·rₓ + pᵧ·rᵧ) )                    (13.1)
```

the minimiser of `Σ‖R(θ)p − r‖²` over rotations about Z. One degree of freedom,
no scale, no translation, no time warp.

The trajectory is **not** modified; `θ̂` is reported; and the resulting position
error is explicitly **blind to a heading error**, so it is a *lower bound* on the
true position error rather than an estimate of it. That is a property of a
dataset in which absolute heading is frequently unobservable, and it is exactly
the kind of thing a Phase 5 filter with a position update would resolve.

---

## 14. What is not compensated

Listed so that a later phase knows what it is inheriting.

**Coning.** Equation (4.1) assumes `ω` is constant across the interval. When the
rotation axis itself rotates within one sample, the attitude accumulates an
error proportional to the enclosed solid angle. A multi-sample coning correction
is the standard remedy.

**Sculling.** The velocity update (7.1) uses `a` at the endpoints. When specific
force and angular rate both vary within an interval and are correlated —
vibration is the classic source — the mean of the endpoints is not the mean over
the interval.

**Midpoint attitude update.** `f_k` is rotated by the attitude *at* `t_k` rather
than at the interval midpoint, a first-order error in the rotation applied to
the specific force.

All three matter when the rotation rate or specific force varies appreciably
*within* one sample interval. At the 10 Hz that dominates this dataset, and
against the attitude error the validation report measures, they are far from the
leading term. The state interface is arranged so they can be added inside
`step()` without changing anything a caller sees.

**Also absent, by scope:** Earth-rate and transport-rate compensation (both
below the sensor noise floor here), the Coriolis term in the velocity equation,
a non-spherical gravity model in the baseline, and any form of aiding — no EKF,
no UKF, no GNSS fusion, no non-holonomic constraint, no zero-velocity
correction, no learned model.

---

## 15. Error propagation, qualitatively

Why an unaided solution cannot be rescued by better integration.

A constant attitude error `δθ` leaks gravity into the horizontal channels:

```text
δa ≈ g₀ · sin(δθ) ≈ 9.81 · δθ                                        (15.1)
```

so 0.6° of tilt error is 0.1 m/s². Constant acceleration error integrates
quadratically:

```text
δv(t) = δa · t              δp(t) = ½ · δa · t²                      (15.2)
```

0.1 m/s² is 6 m/s and 180 m after one minute. A **gyroscope** bias `b_g` is
worse still, because the attitude error it creates grows linearly, making the
position error cubic:

```text
δθ(t) = b_g · t             δp(t) ≈ ⅙ · g₀ · b_g · t³                (15.3)
```

0.01 rad/s (≈0.57°/s) gives ~3.5 km in 60 s — an order of magnitude worse than
the accelerometer bias above, for a perturbation of comparable insignificance.

Nothing in equations (7.1)–(7.2) bounds any of this, and no improvement to the
integration scheme changes the exponent. The baseline's purpose is to make these
rates *measurable* so Phase 5's aiding can be shown to reduce them.

---

## 16. Related documents

- [`../architecture/sensor_frame_conventions.md`](../architecture/sensor_frame_conventions.md)
  — frames, rotation naming, quaternion storage, tolerances.
- [`phase3_orientation.md`](phase3_orientation.md) — the orientation filter and
  alignment estimator this phase consumes.
- [`../datasets/io_vnbd_phase2_contract.md`](../datasets/io_vnbd_phase2_contract.md)
  — what the canonical data guarantees and what it does not.
- [`../../reports/navigation/Phase4_Inertial_Mechanization_Validation.md`](../../reports/navigation/Phase4_Inertial_Mechanization_Validation.md)
  — measured results from a real run.
