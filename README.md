# repr-lab

Representation-learning and benchmark-planning framework for published dataset
releases.

A framework-oriented repository for turning versioned dataset bundles into
deterministic experiment plans, checkpoint-aware benchmark runs, and auditable
experiment artifacts.

## Design Intent

`repr-lab` is designed as a research instrument, not a notebook graveyard.

It is meant to support:

- deterministic experiment identity and artifact layout
- explicit model-family and benchmark-task contracts
- checkpoint provenance that stays visible as executors land
- benchmark planning that hardens before large adapter surfaces accumulate
- reusable training, feature, and analysis seams instead of one-off pipelines

Key principle:

- published dataset bundles, model/task metadata, and experiment artifacts
  should meet through explicit contracts rather than ad hoc notebook state

## Repo Map

- `src/repr_lab/`: active Python runtime for planning, checkpoint manifests,
  feature materialization, execution, training, and analysis
- `tests/`: runtime and CLI validation surface
- `artifacts/`: persisted output surface for planned runs, summaries, and
  derived experiment artifacts
- `artifacts/checkpoints/`: checkpoint provenance and preparation outputs
- `docs/README.md`: docs index
- `docs/architecture.md`: runtime map and benchmark-suite posture
- `docs/runbooks.md`: setup, validation, planning, and doctor reference

## 60-Second Validation

1. Use a Python environment that already contains the repo dependencies.
On declaratively managed workstations, no repo-local `.venv` is required.

2. Sanity-check the environment.

```bash
just setup
```

3. Inspect the public CLI surface.

```bash
PYTHONPATH=src python -m repr_lab.main --version
PYTHONPATH=src python -m repr_lab.main list-models
PYTHONPATH=src python -m repr_lab.main list-tasks
```

4. Run the baseline validation passes.

```bash
just lint
just typecheck
just test
PYTHONPATH=src python -m repr_lab.main doctor
```

## How It Works

```text
Published Release -> Experiment Plan -> Local Checkpoint Manifest + Local
V-JEPA Export Adapter -> Materializer -> Executor -> Benchmark Result
                                          + Run Summary
                                               + Run Report
```

- published-release inspection validates the dataset intake seam
- `plan-release-experiment` writes deterministic `config.json`,
  contract-backed `manifest.json`, and `plan.json` artifacts
- `record-local-checkpoint` normalizes local-only checkpoint provenance without
  claiming automated asset download
- `inspect-vjepa-export` validates and summarizes the bounded repo-local
  V-JEPA raw-media export adapter contract
- `materialize-release-features` turns that bounded local V-JEPA export into
  canonical `FeatureCollection` train/test artifacts inside the planned run
  directory
- `execute-release-plan` consumes that planned run directory as canonical
  config/manifest truth and can either use explicit feature inputs or the
  canonical materialized features already written into that run
- checkpoint preparation records provenance before broader executor paths harden
- executor paths are expected to emit a standard `benchmark_result.json`
  contract plus a typed `run_summary.json` and human-readable
  `analysis/run_report.md`
- downstream consumers should depend only on the bounded run-directory
  contract fields and declared artifact paths rather than on ad hoc internal
  files

## Current Runtime Surfaces

- a contract-aware published-release reader and manifest validator
- a seeded benchmark catalog for V-JEPA 2.1, SAM 3, SAM 2.1, and SAM 1
- explicit benchmark-task metadata for frozen-feature probes, promptable
  segmentation, and video propagation tasks
- a deterministic experiment-planning CLI backed by release-bundle artifacts
- a local-only checkpoint-manifest recorder plus checkpoint-manifest loader
- a bounded repo-local V-JEPA raw-media export adapter contract with a legacy
  metadata bridge for older local feature batches
- a runnable V-JEPA 2.1 materialization-plus-execution path that turns those
  bounded local exports into canonical feature collections and then emits
  benchmark artifacts
- a bounded `repr_lab_run_directory_contract@1.0.0` across `manifest.json`
  and `benchmark_result.json` for downstream figure-oriented readers
- typed run-summary and run-report artifacts for the first runnable benchmark
  path
- checkpoint provenance, execution-maturity tracking, and a `doctor` readiness
  surface that makes local-only paths and optional raw-media dependencies
  explicit
- a trainer, feature-store, and analysis spine that now backs the first
  runnable benchmark path

## Current Focus

- keep the first runnable V-JEPA 2.1 materialization-plus-execution path
  bounded and deterministic before broadening the adapter surface
- keep the local raw-media export adapter explicit without promoting it into a
  shared downstream contract
- keep the evaluator and reporting layer explicit before claiming a generalized
  raw-media adapter runtime
- keep benchmark compatibility explicit across modality, prompt, and dataset
  interfaces
- preserve checkpoint provenance as real weights and executors start landing

## Safety Posture

- preserve upstream dataset-release provenance in experiment artifacts
- make execution maturity explicit instead of implying every catalog entry is
  runnable
- record exact checkpoint sources and revisions once concrete adapters land
- keep historical artifacts additive and auditable rather than silently
  rewriting them

## Docs Map

- `docs/README.md`: docs index
- `docs/architecture.md`: architecture, benchmark posture, and artifact model
- `docs/runbooks.md`: setup, validation, planning, checkpoint, and doctor
  reference
- `artifacts/README.md`: artifact-output guidance

## License

This project is licensed under the MIT License. See `LICENSE` for the full
text.
