# Docs Index

Use this folder for:
- architecture and benchmark-runtime references
- setup, validation, checkpoint-manifest, materialization, execution, and
  reporting runbooks
- runtime notes that explain the current experiment and benchmark spine

Active runtime note:

- `src/repr_lab/` is the live Python runtime tree
- the current public CLI includes `inspect-release`, `list-models`,
  `list-tasks`, `plan-release-experiment`, `record-local-checkpoint`,
  `inspect-vjepa-export`, `materialize-release-features`,
  `execute-release-plan`,
  `prepare-checkpoint`, and `doctor`
- runtime artifacts and persisted outputs live under `../artifacts/`
- the first runnable path now emits `benchmark_result.json`,
  `run_summary.json`, and `analysis/run_report.md` inside each planned run

Validation note:

- the default validation commands are `just lint`, `just typecheck`, and
  `just test`
- `python -m repr_lab.main doctor` is the readiness check for benchmark
  automation, local-only-path posture, and optional raw-media runtime blockers
- `docs/runbooks.md` is the detailed setup and validation guide

Current high-value docs:

- `docs/architecture.md`
- `docs/runbooks.md`
