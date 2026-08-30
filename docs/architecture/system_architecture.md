# System Architecture

High-level data/control flow for the eventual system (SIH26168). This
documents the established architectural intent — it does not describe
anything implemented yet beyond Phase 0 scaffolding.

```text
Data
 ↓
ML / Research   (Python: python/idr/)
 ↓
Navigation Core (C++: core/)
 ↓
Android / Edge  (not yet implemented)
```

## Layers

- **Data** — smartphone IMU (accelerometer, gyroscope, magnetometer) and
  GNSS logs, plus reference/ground-truth datasets (e.g. IO-VNBD) for
  offline development. See `data/README.md`.
- **ML / Research (Python)** — dataset handling, model training/evaluation,
  experimentation. Lives under `python/idr/`. Produces exported model
  artifacts (`models/exported/`) consumed by the navigation core; does not
  itself run on-device.
- **Navigation Core (C++)** — the reusable inertial-navigation /
  sensor-fusion engine. Lives under `core/`. Deployable in four contexts
  without modification to its public interface: dataset replay, desktop
  testing, Android (via a thin adapter), and external-IMU edge devices.
- **Android / Edge** — thin platform adapters that feed sensor data into
  the navigation core and surface its output. Not started; see
  [`docs/sih/phase0_scope.md`](../sih/phase0_scope.md) for what's deferred.

## Why this shape

Keeping the navigation core a standalone C++ library (Rule 5 in
[`engineering_rules.md`](../development/engineering_rules.md)) means the
same estimator can be validated against recorded datasets on a desktop
long before an Android build exists, and later reused unchanged on an
external-IMU edge target.

See [`software_architecture.md`](software_architecture.md) for how this
maps onto the actual repository structure.
