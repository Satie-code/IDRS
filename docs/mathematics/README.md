# Mathematics

Derivations and reference material for the navigation math.

- [`phase3_orientation.md`](phase3_orientation.md) — quaternion and
  rotation-matrix algebra, the gravity estimator, the complementary orientation
  filter, magnetometer quality, and the phone→vehicle alignment estimator, as
  implemented in [`python/idr/frames/`](../../python/idr/frames/). The
  conventions those equations assume are fixed in
  [`../architecture/sensor_frame_conventions.md`](../architecture/sensor_frame_conventions.md).
- [`phase4_mechanization.md`](phase4_mechanization.md) — the strapdown
  mechanization: specific force and the `a = f + g` sign, the gravity model,
  quaternion attitude propagation, the frame chain into the navigation frame,
  trapezoidal velocity and position integration, the time-step policy, what a
  stationary window can and cannot determine about bias, the residual-lag
  estimator, and the terms deliberately not compensated. Implemented in
  [`python/idr/navigation/`](../../python/idr/navigation/).

Still to come, in the phases that implement them: EKF/UKF formulations,
non-holonomic constraints, GNSS/INS fusion, and learned correction. Nothing is
documented here before it exists — see
[`../development/engineering_rules.md`](../development/engineering_rules.md),
Rule 10.
