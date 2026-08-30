# Models

```text
models/
├── checkpoints/  # Training checkpoints. Gitignored — regenerable, often large.
├── exported/     # Exported inference artifacts (e.g. ONNX). Gitignored — large binaries.
└── metadata/     # Small, hand-written docs about models (architecture notes,
                  # training config summaries, version notes). Tracked in Git.
```

Same rationale as `data/README.md`: large binary artifacts don't belong in
Git history. Nothing is produced into this directory in Phase 0 — no model
training happens yet (see `docs/sih/phase0_scope.md`).
