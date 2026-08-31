# Mathematics

Derivations and reference material for the navigation math.

- [`phase3_orientation.md`](phase3_orientation.md) — quaternion and
  rotation-matrix algebra, the gravity estimator, the complementary orientation
  filter, magnetometer quality, and the phone→vehicle alignment estimator, as
  implemented in [`python/idr/frames/`](../../python/idr/frames/). The
  conventions those equations assume are fixed in
  [`../architecture/sensor_frame_conventions.md`](../architecture/sensor_frame_conventions.md).

Still to come, in the phases that implement them: strapdown INS mechanization,
EKF/UKF formulations, non-holonomic constraints, and GNSS/INS fusion. Nothing
is documented here before it exists — see
[`../development/engineering_rules.md`](../development/engineering_rules.md),
Rule 10.
