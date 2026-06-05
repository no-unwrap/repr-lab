# Runbooks

## Purpose

Technical guide for setup, validation, debugging, and runtime notes.

## Local Environment Source

- the local environment file lives at the repo root as `.env`
- keep `.env.example` placeholder-only and never commit `.env`
- recommended pattern:

```bash
set -a
source .env >/dev/null 2>&1
set +a
```

## Setup

1. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
just setup
```

Convenience recipes:
- `just setup`
- `just lint`
- `just typecheck`
- `just test`

3. Check the public CLI surface.

```bash
python -m repr_lab.main --version
python -m repr_lab.main list-models
python -m repr_lab.main list-tasks
python -m repr_lab.main record-local-checkpoint --help
python -m repr_lab.main inspect-vjepa-export --help
python -m repr_lab.main materialize-release-features --help
python -m repr_lab.main execute-release-plan --help
```

## Baseline Validation

Use this sequence after meaningful runtime or documentation changes:

```bash
python -m ruff check .
python -m mypy src
python -m pytest -q
```

## Benchmark Catalog

List the currently tracked benchmark model families:

```bash
python -m repr_lab.main list-models
python -m repr_lab.main describe-model --model vjepa2
python -m repr_lab.main describe-model --model sam3
python -m repr_lab.main describe-model --model sam2
python -m repr_lab.main describe-model --model sam1
```

List benchmark tasks:

```bash
python -m repr_lab.main list-tasks
python -m repr_lab.main list-tasks --model sam2
python -m repr_lab.main list-tasks --model sam3
python -m repr_lab.main describe-task --task promptable-mask-refinement
```

SAM 3 note:

- checkpoints are gated on Hugging Face at `facebook/sam3`
- use the authenticated `hf` / `huggingface-cli` session on this machine when pulling them
- record the exact checkpoint revision or snapshot path in experiment extras once a concrete adapter lands

## Checkpoint Preparation

Automated checkpoint preparation is currently implemented for Hugging Face-backed
families.

Example:

```bash
python -m repr_lab.main prepare-checkpoint --model sam3
```

This writes checkpoint provenance to:

- `artifacts/checkpoints/sam3/sam3/checkpoint.json`

The current seeded families expose explicit provenance, but only SAM 3 has an
automated preparation path today. V-JEPA 2.1, SAM 2.1, and SAM 1 remain
cataloged with manual or upstream-repo checkpoint flows until broader automated
adapters land.

For those local-only families, normalize the resolved checkpoint state into a
manifest before materialization:

```bash
python -m repr_lab.main record-local-checkpoint \
  --model vjepa2 \
  --local-path /local/path/vjepa2_snapshot \
  --output /local/path/vjepa2_checkpoint_manifest.json \
  --file model.pt
```

## Compatibility Contracts

The benchmark catalog now records:

- model-family input modalities such as `image` and `video`
- model-family prompt capabilities such as `spatial`, `text`, `exemplar`, and `initial_mask`
- task dataset interfaces such as `published_release` and `video_sequence`

That metadata is part of experiment planning, so incompatible pairings are
rejected before executor code lands.

## Doctor

Check local benchmark readiness:

```bash
python -m repr_lab.main doctor
```

This reports:

- whether `huggingface-cli` / `hf` is available
- whether the current shell is logged into Hugging Face
- whether Hugging Face-backed model families are actually ready for automated
  checkpoint preparation
- which runnable families still use intentionally local-only reference paths
- whether optional raw-media runtime packages such as `torch`,
  `torchvision`, `PIL`, and `transformers` are installed for the bounded local
  raw-media adapter contract

## Benchmark Result Contract

Executor paths should emit a standard benchmark record at:

- `benchmark_result.json`

inside each planned run directory. The runtime now provides a standard
`BenchmarkResultRecord` surface for that output so future adapters do not invent
their own result shapes.

The bounded downstream experiment seam is now the pair:

- `manifest.json`
- `benchmark_result.json`

Both files carry the `repr_lab_run_directory_contract@1.0.0` headers so
downstream readers can validate the run-directory boundary without depending on
undeclared internal files.

The first runnable path also emits:

- `artifacts/run_summary.json`
- `analysis/run_report.md`

`run_summary.json` is the compact typed contract for downstream tooling and
audits. `analysis/run_report.md` is the operator-facing narrative view over the
same run, including metrics, provenance, warnings, and artifact links.

## Bundle-Backed Planning

Create an experiment-ready plan from a published release bundle:

```bash
python -m repr_lab.main plan-release-experiment \
  --bundle-dir /path/to/release_bundle \
  --model vjepa2 \
  --task frozen-feature-localization-probe
```

This writes:

- `config.json`
- `manifest.json`
- `plan.json`

under a deterministic run directory in `artifacts/planned/`.

Use SAM families for segmentation-oriented plans:

```bash
python -m repr_lab.main plan-release-experiment \
  --bundle-dir /path/to/release_bundle \
  --model sam2 \
  --task promptable-mask-refinement
```

## Runnable Frozen Feature Probe

The first runnable executor path is:

- model family: `vjepa2`
- task: `frozen-feature-localization-probe`

It consumes a previously planned run directory plus either:

- a local feature export batch and local checkpoint manifest, which repr-lab
  then materializes into canonical `train` and `test` `FeatureCollection`
  artifacts inside the run directory
- or explicit `train` and `test` feature collections when those already exist

The local feature export batch is a compressed `.npz` governed by the
repo-local `repr_lab_vjepa_local_raw_media_export@0.1.0` adapter contract. It
carries:

- one or more layer arrays
- a 1D `sample_ids` array aligned with those layers
- `sample_id_kind` of `asset_id` or `file_name`
- export metadata including `model_variant`
- explicit-contract metadata including `adapter_contract_name`,
  `adapter_contract_version`, and `adapter_name`
- optional export metadata such as `default_layer`

Inspect or validate a local export before materialization:

```bash
python -m repr_lab.main inspect-vjepa-export \
  --feature-export /local/path/vjepa2_export.npz
```

Legacy `LocalFeatureExport` batches that only carry V-JEPA metadata are still
accepted in compatibility mode so older local experiments can be rematerialized
without rewriting the shared run-directory contract.

Required feature-collection metadata:

- `release_id`
- `release_version`
- `model_family`
- `model_variant`

The metadata should also carry local checkpoint provenance when available so the
run artifacts preserve how the features were extracted.

Example:

```bash
python -m repr_lab.main materialize-release-features \
  --plan artifacts/planned/<run-id>/plan.json \
  --feature-export /local/path/vjepa2_export.npz \
  --checkpoint-manifest /local/path/vjepa2_checkpoint_manifest.json

python -m repr_lab.main execute-release-plan \
  --plan artifacts/planned/<run-id>/plan.json
```

If you already have canonical feature collections, you can still bypass the
materializer and pass them explicitly:

```bash
python -m repr_lab.main execute-release-plan \
  --plan artifacts/planned/<run-id>/plan.json \
  --train-features /local/path/vjepa2_train_features.npz \
  --test-features /local/path/vjepa2_test_features.npz
```

The materialization-plus-execution flow writes benchmark artifacts back into the
planned run directory, including:

- `checkpoints/resolved_checkpoint.json`
- `artifacts/materialized_features_train.npz`
- `artifacts/materialized_features_test.npz`
- `artifacts/feature_materialization.json`
- copied canonical feature stores under `artifacts/`
- `artifacts/probe_metrics.json`
- `artifacts/feature_sources.json`
- `artifacts/run_summary.json`
- `artifacts/linear_probe.npz`
- `analysis/confusion_summary.json`
- `analysis/layerwise_metrics_train.json`
- `analysis/layerwise_metrics_test.json`
- `analysis/run_report.md`
- `benchmark_result.json`

Current reference-path note:

- the runnable V-JEPA path is still explicitly local-only at the checkpoint
  and raw-feature boundary
- the explicit V-JEPA export adapter contract is repo-local and does not widen
  the downstream shared experiment contract
- `materialize-release-features` is a normalization step, not a claim that
  repr-lab already owns a generalized raw-media adapter runtime
- `doctor` will report that posture directly so later tranches do not mistake
  the current executor seam for broader automation than is actually landed

## Focused Test Runs

Examples:

```bash
python -m pytest -q tests/test_cli.py
python -m pytest -q tests/test_config.py
python -m pytest -q tests/test_trainer.py
```

## Extension Seams

- implement dataset-release readers that consume published release bundles
- add model adapters and backend-specific trainer implementations on top of the seeded benchmark catalog
- extend analyzers and report exporters for benchmark tasks built on published datasets
- keep runtime contracts centered on published dataset and experiment artifacts
