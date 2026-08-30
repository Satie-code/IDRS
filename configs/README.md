# Configuration

Environment-specific configuration files, loaded by `idr.config.load_config`
(see `python/idr/config/`).

## Files

- `development.yaml` — local development defaults.
- `testing.yaml` — used by the test suite when it needs a file-backed config
  (most tests pass `config_dir` explicitly instead; see `tests/python/test_config.py`).
- `production.yaml` — not yet created. Add it when there is an actual
  deployment target; don't speculatively invent production values now.

## How resolution works

1. Dataclass defaults in `idr.config.schema`.
2. `configs/<environment>.yaml`, if present.
3. Environment variables prefixed `IDR_`, using `__` for nesting
   (e.g. `IDR_LOGGING__LEVEL=DEBUG` overrides `logging.level`).

The active environment is chosen by an explicit argument, else the
`IDR_ENVIRONMENT` environment variable, else `development`.

## Phase 0 scope

Only the settings the foundation itself needs (environment name, logging)
are defined here. Sensor rates, model paths, navigation/map settings, etc.
belong to the phases that implement those subsystems — don't add
placeholder values for them now; an empty section here is not "more done"
than no section at all, and invented defaults would just have to be
re-validated later anyway.
