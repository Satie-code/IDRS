# Engineering Rules (All Phases)

These apply for the lifetime of the project, not just Phase 0. Referenced
from the root `README.md`; keep this the single copy.

1. **Never modify raw datasets.** `data/raw/` (and any local dataset copy,
   e.g. `data/IO-VNBD_dataset/`) is read-only input. Derived data goes in
   `data/intermediate/` or `data/processed/`, never back into raw.
2. **Never hardcode environment-specific paths.** Resolve paths relative to
   the repo root (`idr.utils.find_repo_root`) or via configuration, not
   absolute strings like `C:\Material\Repo\IDRS\...`.
3. **Do not couple UI code to navigation mathematics.** Android/edge
   adapters call into the navigation core through a stable interface; they
   never reimplement or reach into its internals.
4. **Do not couple ML training code to runtime inference.** Training
   (Python, offline) and inference (on-device, latency-sensitive) are
   separate concerns with separate code paths, joined only by an exported
   model artifact.
5. **Navigation Core must remain independent of Android.** `core/` must
   build and run standalone (desktop, dataset replay, external-IMU edge)
   with zero Android/JNI dependencies in the core library itself.
6. **All important algorithmic changes must have tests.** See
   [`testing.md`](testing.md).
7. **Benchmark results must be reproducible.** Pinned inputs, recorded
   environment/config, checked-in benchmark code — not one-off numbers in a
   chat message.
8. **Never claim performance that has not been experimentally measured.**
   No "should be about Xhz" or "probably accurate to Y meters" without a
   benchmark or test backing the number.
9. **Training must never leak future information into inference/evaluation.**
   No target leakage, no evaluating on data used for fitting, no using
   future timestamps to inform a causal, real-time estimator.
10. **Do not start future phases early.** Phase 0 is foundation only — no
    INS/EKF/UKF, GNSS fusion, map matching, NHC, AI models, Android app, or
    ONNX integration. Each phase is scoped and implemented deliberately.
