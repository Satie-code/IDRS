# Sensor Frame Conventions

The frames, axis orders, rotation conventions and numerical tolerances that
everything in [`python/idr/frames/`](../../python/idr/frames/) obeys. The
machine-readable version is
[`idr.frames.conventions`](../../python/idr/frames/conventions.py); this
document explains the choices. The equations are in
[`../mathematics/phase3_orientation.md`](../mathematics/phase3_orientation.md).

Most orientation bugs are convention bugs. A library that is internally
self-consistent under the wrong convention passes every round-trip test it has
and still rotates the data backwards. So the conventions are written down,
enforced in the type system where possible, and asserted by tests that would
fail if a convention silently flipped.

## 1. Navigation frame — ENU

```text
X = East
Y = North
Z = Up
```

Right-handed local tangent plane. Gravity points along **−Z**.

Chosen over NED because the vehicle frame here has Z up, and using an
up-positive navigation frame means the vertical axis has one sign throughout
the stack rather than flipping at the vehicle/navigation boundary. Sign flips
at frame boundaries are exactly the kind of error that survives review. NED is
common in aerospace; nothing in this project needs it, and mixing the two would
be worse than either.

The origin is a local tangent point, not fixed here. Phase 3 never converts
latitude and longitude into a navigation-frame position — that is Phase 4's
job, and it will need to state the origin and the projection it uses.

## 2. Vehicle frame

```text
X = forward
Y = left
Z = up
```

Right-handed. With X forward and Z up, the third axis is **left**, not right —
"lateral" alone is ambiguous about sign, so the sign is fixed here and referred
to from everywhere else.

The consequences, which are what the code and its tests actually assert:

- A vehicle accelerating forward shows **positive** specific force on X.
- A vehicle turning **left** has a positive yaw rate about +Z, and the
  centripetal specific force appears as **positive** Y.
- A stationary vehicle reads **+g on Z**, not −g and not zero, because an
  accelerometer measures specific force including the reaction to gravity.

That last point is worth stating plainly, because it is the sign that a
synthetic fixture will happily get wrong and thereby hide a real defect.

## 3. Phone / sensor frame — and what is genuinely unknown

Canonical channels are `accel_x/y/z`, `gyro_x/y/z`, `mag_x/y/z`,
`gravity_x/y/z`, established by Phase 2 **by column position**.

**The physical direction of those axes is not established by this dataset.**
That is a real gap, not a formality, and it is worth separating three things:

1. **The Android runtime convention.** For a device held upright facing the
   user: X points right, Y points toward the top of the screen, Z points out of
   the screen toward the user. The accelerometer and the gravity sensor both
   report specific force, so a device lying flat and screen-up reads
   approximately `(0, 0, +9.81)`.

2. **What IO-VNBD actually contains.** The dataset does not state its axis
   convention. Phase 1 found no documentation of it, and Phase 2 found that the
   gyroscope columns are labelled `Yaw/Pitch/Roll` in 158 files and `X/Y/Z` in
   73 — with byte-identical data rows. The labels are therefore aliases and
   carry no axis semantics.

3. **What Phase 3 verified.** One thing, and only one: the accelerometer and
   the gravity channel agree in sign. Across 66 real segments with a
   quasi-static interval, the median cosine between them is **+0.999**, and the
   gravity channel's magnitude is exactly 9.807 m/s². Both read specific force,
   consistent with the Android convention on that point.

Nothing above establishes that phone X is physically "right". **Do not assume
it.** Every algorithm in this phase is invariant to the question, because the
phone→vehicle rotation is estimated from the data rather than assumed — which
is also what the deployed system needs, since a phone in a cupholder has no
fixed relationship to the vehicle at all.

## 4. Rotation naming — source and destination, always

A rotation is named for the two frames it connects, **destination first**:

```text
v_vehicle = R_vehicle_phone @ v_phone
v_navigation = R_navigation_vehicle @ v_vehicle
```

so composition chains left to right and the inner frames cancel visually:

```text
R_navigation_phone = R_navigation_vehicle @ R_vehicle_phone
```

There is no `R1` or `R2` anywhere in the package. The direction of a rotation
is precisely the thing that gets silently inverted, and a name that does not
carry it makes the inversion invisible.

This is enforced, not merely documented.
[`FrameRotation`](../../python/idr/frames/rotations.py) carries its two frames;
applying it to a vector tagged with a different frame raises, and composing two
rotations whose inner frames do not meet raises rather than returning a
plausible wrong matrix.

### The formal statement

If frame B's basis vectors are frame A's basis actively rotated by `R`, then a
vector's coordinates convert as `v_A = R @ v_B`. So `R_A_B` is simultaneously
"the active rotation carrying A's basis onto B's" and "the map from
B-coordinates to A-coordinates". Both readings are the same matrix; conflating
them is safe, and only *this* pairing is.

## 5. Rotation representations

| Representation | Used for | Notes |
|---|---|---|
| Quaternion | The internal representation | Hamilton, stored `[x, y, z, w]` |
| Rotation matrix | Transforms, SO(3) validation | Columns are the destination basis |
| Euler angles | **Diagnostics only** | Z-Y-X intrinsic |

### Quaternions

- **Storage order `[x, y, z, w]`** — vector part first, scalar last. SciPy uses
  this order; Eigen and many textbooks use `[w, x, y, z]`. Mixing them silently
  substitutes one rotation for another, so `QUATERNION_SCALAR_INDEX` names the
  scalar slot rather than leaving it to memory.
- **Hamilton product**, not JPL: `ij = k`.
- **Active rotations.** `rotate_vector(q, v)` rotates `v` within one fixed
  frame, right-handed about the axis. `from_axis_angle(axis, θ)` rotates
  vectors by `+θ`.
- **`q` and `−q` are the same rotation.** Every comparison in the package is
  sign-agnostic, and no function may report the two as different.

### Euler angles are diagnostics, never state

Z-Y-X intrinsic: `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`. Roll is applied first
about body X, then pitch about the new Y, then yaw about the new Z.

They are never the internal representation, because they are discontinuous,
sequence-dependent, and degenerate at |pitch| = 90°, where only `yaw − roll`
is defined and the individual angles are not. `to_euler` resolves that case by
assigning everything to yaw — a *choice*, not a measurement — so
`euler_is_well_conditioned(q)` exists and must be checked before quoting a
roll/yaw split.

## 6. Numerical tolerances

Each guards a specific operation in float64. A single blanket `1e-6` would be
both too loose for a quaternion norm and too tight for an Euler round trip.

| Constant | Value | What it guards |
|---|---|---|
| `QUATERNION_NORM_TOLERANCE` | `1e-9` | `‖q‖ − 1`. A norm is 4 squares, a sum and a root — about 1e-15 of error. |
| `ORTHONORMALITY_TOLERANCE` | `1e-9` | `max\|RᵀR − I\|`. Nine dot products of three terms each. |
| `DETERMINANT_TOLERANCE` | `1e-9` | `det(R) − 1`. Separates a rotation from a reflection. |
| `ROUND_TRIP_TOLERANCE` | `1e-12` | quaternion→matrix→quaternion. Measured worst case 8.7e-16 rad. |
| `EULER_GIMBAL_LIMIT_RAD` | `1e-7` | How close \|pitch\| may come to 90° before roll/yaw stop separating. |
| `MIN_VECTOR_NORM` | `1e-12` | Below this a vector has no direction and normalizing it amplifies noise. |

**The determinant check is not redundant.** An orthonormal matrix with
`det = −1` is a reflection: it satisfies `RᵀR = I`, passes any orthonormality
test, and quietly mirrors the data — turning a left turn into a right one.

One tolerance changed during Phase 3, for a reason worth keeping.
`ROUND_TRIP_TOLERANCE` was `1e-8` while `angular_distance` computed
`2·acos(|a·b|)`. `acos` has an infinite derivative at 1, so two rotations
differing by a single ulp measured as ~1e-8 apart — a floor belonging to the
*measurement*, not to the round trip. Rewriting it as
`2·atan2(‖vec(a⁻¹b)‖, |w(a⁻¹b)|)`, which is well conditioned near zero, dropped
the measured worst case to 8.7e-16 and the tolerance was tightened by four
orders of magnitude to match.

## 7. Gravity, and the sign that matters

An accelerometer measures **specific force**, not gravity:

```text
f = a − g          (g points down; f is what the sensor reports)
```

At rest `a = 0`, so `f = −g` and the sensor reads *upward*. Consequently:

- `GravityEstimate.direction` is the **gravity vector**, pointing **down**.
  Both source channels must be negated to obtain it.
- Recovering linear acceleration means `a = f + g` — the gravity vector is
  **added**, not subtracted. Subtracting it doubles the vertical term instead
  of cancelling it, and the result still looks like an acceleration.

Both of these were live defects during Phase 3, caught by a synthetic round
trip and by a unit test respectively. Neither produced an obviously wrong
number; the first produced a 180° heading error.

## 8. Frame-tagged vectors

`Vector3` and `VectorSeries` carry a `Frame`. Operations across mismatched
frames raise instead of computing. This prevents the mundane, expensive failure
of passing phone-frame data into something expecting vehicle-frame data and
getting plausible numbers back.

This is deliberately **not** a units framework. There is no dimensional
analysis and no unit arithmetic — Phase 2 already normalizes units on ingest
and records them in `canonical_schema.json`. What is tracked is the frame, the
timestamp and the provenance, because those are the three things that were
actually ambiguous in this dataset.

## 9. Related documents

- [`../mathematics/phase3_orientation.md`](../mathematics/phase3_orientation.md)
  — the equations, the filter, and the alignment estimator.
- [`../datasets/io_vnbd_phase2_contract.md`](../datasets/io_vnbd_phase2_contract.md)
  — what the canonical data guarantees and what it does not.
- [`../../reports/sensor_alignment/Phase3_Sensor_Frame_Validation.md`](../../reports/sensor_alignment/Phase3_Sensor_Frame_Validation.md)
  — measured results from a real run.
