# SIH26168 — Phase 0 Scope Record

Problem: **AI-ML based Intelligent Dead Reckoning System for Seamless
Navigation** (ISRO, Smart India Hackathon). The end system must maintain
smartphone navigation continuity during GNSS outages using IMU data,
AI/ML-based motion estimation, inertial navigation, non-holonomic
constraints, offline map matching, and GNSS+INS fusion — targeting roughly
10 Hz smartphone navigation updates.

## What Phase 0 delivered

Engineering foundation only: repository structure, Python packaging,
C++/CMake scaffolding, testing infrastructure (pytest + CTest), a typed
configuration system, standard-library-based logging, code-quality tooling
(Ruff, mypy, pre-commit), a CI skeleton, and documentation. See the root
`README.md` for the full breakdown and `docs/development/` for setup and
standards.

## What Phase 0 explicitly did NOT implement

INS, EKF/UKF, quaternion navigation, GNSS+INS fusion, map matching,
non-holonomic constraints, AI/ML models (no PyTorch/TensorFlow/ONNX
dependency added), the Android application, or any dead-reckoning
algorithm. No IO-VNBD dataset download, inspection, or preprocessing was
performed as part of this phase, even though a local copy of the dataset
already exists on this machine under `data/IO-VNBD_dataset/` — it was left
untouched (see `data/README.md`).

## Why this boundary

Rule 10 in
[`../development/engineering_rules.md`](../development/engineering_rules.md):
do not start future phases early. Building navigation logic on top of an
unverified foundation risks having to redo both. Each later phase (dataset
ingestion, sensor processing, INS, GNSS fusion, map matching, on-device
inference, Android integration) gets its own scoped Claude Code session.
