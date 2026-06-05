# Architecture

## Current Runtime Map

- `src/repr_lab/config.py`
  - deterministic experiment config identity, deep overrides, and grid expansion
- `src/repr_lab/experiment.py`
  - standardized run-directory layout plus manifests, feature paths, and result paths
- `src/repr_lab/dataset.py`
  - published-dataset metadata contract for reproducible loader registration
- `src/repr_lab/published_dataset.py`
  - contract-aware release reader and manifest validation surface
- `src/repr_lab/model_catalog.py`
  - first-class benchmark model and task catalog for external backbone families
- `src/repr_lab/checkpointing.py`
  - checkpoint provenance, local checkpoint-manifest handling, execution
    maturity, and model-asset preparation helpers
- `src/repr_lab/materialization.py`
  - bounded local V-JEPA export adapter contract, legacy metadata bridge, and
    release-aligned feature materialization into canonical run-local
    `FeatureCollection` artifacts
- `src/repr_lab/execution.py`
  - planned-run execution loader plus the first runnable V-JEPA 2.1 frozen-feature probe path
- `src/repr_lab/benchmark_result.py`
  - standard benchmark-result contract for executor paths
- `src/repr_lab/reporting.py`
  - typed run-summary contract plus human-readable run-report renderer
- `src/repr_lab/planning.py`
  - published-release experiment planning surface that binds bundle provenance to model/task choices
- `src/repr_lab/trainer.py`
  - backend-agnostic training loop with multi-run aggregation and best-metric tracking
- `src/repr_lab/features.py`
  - compressed persisted layer-feature format for later analysis
- `src/repr_lab/analysis.py`
  - layerwise analyzer surface plus built-in representation metrics
- `src/repr_lab/registry.py`
  - named registry abstraction for future datasets, models, trainers, and analyzers
- `src/repr_lab/main.py`
  - public CLI surface for versioning and future experiment entrypoints

## Current Architectural Posture

The runtime is intentionally small but real. It already owns the minimum viable research spine for deterministic experiment definition, stable artifact layout, feature persistence, layerwise analysis, and benchmark-family intake.

## Runtime Shape

`repr-lab` focuses on versioned dataset-release loading, benchmarking,
training runs, and experiment analysis.

The runtime works from:
- versioned release bundles as dataset inputs
- stable manifests and artifact directories for experiment outputs, with
  `manifest.json` plus `benchmark_result.json` now carrying an explicit
  bounded run-directory contract for downstream readers

The runtime is no longer just scaffolding. It now has the minimum viable research spine:

- configs can be versioned, overridden, and swept deterministically
- runs get stable manifests and result files
- benchmark model families and benchmark tasks are explicit runtime objects instead of implicit notebook lore
- checkpoint provenance and execution maturity are explicit runtime concepts instead of side notes in docs
- task compatibility is expressed in terms of image/video modality and prompt interface, not only family-name allowlists
- trainers can orchestrate repeated trials without binding the repo to one backend
- saved features can feed reusable layerwise analyzers

This is intentionally still lean. The current code establishes the key runtime
contracts without dragging along monolithic dataset/model/trainer coupling.

## Extension Seams

As the runtime expands, keep new work grouped around these seams:

- dataset registries plus concrete loaders
- benchmark model families, model factories, and checkpoint contracts
- PyTorch-first trainer implementations on top of `TrainerBase`
- richer analyzer libraries and report exporters
- experiment launch orchestration across configs, runs, and post-hoc analysis

## Benchmark Suite Direction

`repr-lab` is now oriented toward becoming a benchmark suite rather than a loose
experiment scratchpad.

The current seeded model families are:

- `vjepa2`
  - representation-learning and temporal anticipation backbone family
- `sam3`
  - open-vocabulary concept segmentation and tracking baseline
- `sam2`
  - promptable image/video segmentation and propagation baseline
- `segment-anything`
  - static-image promptable segmentation baseline with ONNX-exportable decoder

The current benchmark-task catalog covers:

- frozen-feature localization probes
- automatic mask proposal
- promptable mask refinement
- open-vocabulary concept segmentation
- video mask propagation
- open-vocabulary concept tracking
- action anticipation

The runtime does not vendor upstream model repos. Instead, it tracks explicit
family metadata, task compatibility, runtime requirements, and experiment-plan
artifacts so future adapters can land without losing provenance.

The first runnable executor path now treats the planned run directory as the
canonical execution boundary. It can normalize a local-only checkpoint manifest,
consume a bounded repo-local V-JEPA raw-media export adapter payload, bridge
older metadata-only local exports when needed, materialize plan-matched
train/test `FeatureCollection` artifacts into the run directory, and then run
the lightweight NumPy probe on either those canonical materialized features or
explicit external feature collections. The optional raw-media runtime packages
remain a repo-local readiness contract rather than a generalized shared adapter
claim. The path still writes copied canonical inputs, layerwise analysis,
confusion artifacts, the standard `benchmark_result.json` record, a typed
`run_summary.json`, and a human-readable `analysis/run_report.md`.

That run directory is now also the bounded downstream experiment contract for
`fig-lab`: consumers should rely on the explicit contract headers in
`manifest.json` and `benchmark_result.json` plus declared `artifact_paths`,
not on undeclared internal files.

It now also distinguishes between:

- `catalog_only`
- `planned`
- `runnable`
- `validated`

for both model families and benchmark tasks, so the suite can scale without
pretending every catalog entry is already executable.

## Artifact Direction

`artifacts/` is the persisted output surface for:

- experiment configs
- run manifests
- aggregated result summaries
- checkpoints
- extracted features
- analysis outputs
- compact run summaries
- human-readable run reports

Historical artifacts should be additive and auditable rather than silently overwritten.
