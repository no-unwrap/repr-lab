from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repr_lab._parsing import coerce_dict, load_json_object
from repr_lab.config import ExperimentConfig
from repr_lab.experiment import ExperimentPaths
from repr_lab.model_catalog import (
    BenchmarkTaskSpec,
    ModelFamilySpec,
    ModelVariantSpec,
    resolve_benchmark_task,
    resolve_model_family,
)
from repr_lab.published_dataset import PublishedRelease, load_published_release
from repr_lab.run_contract import validate_run_manifest_payload


@dataclass(frozen=True, slots=True)
class PlannedReleaseExperiment:
    release: PublishedRelease
    task: BenchmarkTaskSpec
    model_family: ModelFamilySpec
    model_variant: ModelVariantSpec
    config: ExperimentConfig
    paths: ExperimentPaths

    def to_dict(self) -> dict[str, Any]:
        return {
            "release": self.release.to_dict(),
            "task": self.task.to_dict(),
            "model_family": self.model_family.to_dict(include_variants=False),
            "model_variant": self.model_variant.to_dict(),
            "execution_maturity": {
                "model_family": self.model_family.maturity.value,
                "benchmark_task": self.task.maturity.value,
            },
            "config": self.config.to_dict(),
            "paths": {
                "run_dir": str(self.paths.run_dir),
                "config_path": str(self.paths.config_path),
                "manifest_path": str(self.paths.manifest_path),
                "benchmark_result_path": str(self.paths.benchmark_result_path),
                "plan_path": str(self.paths.plan_path),
            },
        }

    def write(self) -> "PlannedReleaseExperiment":
        self.paths.initialize(self.config)
        self.paths.plan_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self

    @classmethod
    def load(cls, path: str | Path) -> "PlannedReleaseExperiment":
        return load_planned_release_experiment(path)


def plan_release_experiment(
    bundle_dir: str | Path,
    *,
    model_family: str,
    task: str,
    variant: str | None = None,
    experiment_name: str | None = None,
    artifacts_root: str | Path = "artifacts/planned",
    seed: int = 0,
) -> PlannedReleaseExperiment:
    release = load_published_release(bundle_dir)
    resolved_family = resolve_model_family(model_family)
    resolved_task = resolve_benchmark_task(task)
    resolved_task.validate_model_family(resolved_family)
    resolved_task.validate_release(release)
    resolved_variant = resolved_family.resolve_variant(variant)

    name = experiment_name or f"{release.release_id}-{resolved_task.name}-{resolved_family.name}"
    config = ExperimentConfig(
        name=name,
        dataset=release.release_id,
        model=resolved_variant.name,
        seed=seed,
        tags=("published-release", resolved_task.name, resolved_family.name),
        extras={
            "release": {
                "release_id": release.release_id,
                "release_version": release.release_version,
                "publisher_repo": release.publisher_repo,
                "publisher_commit_sha": release.publisher_commit_sha,
                "bundle_dir": str(release.bundle_dir),
                "task_types": list(release.task_types),
                "source_formats": list(release.source_formats),
                "counts": release.counts,
            },
            "benchmark_task": resolved_task.to_dict(),
            "model_family": resolved_family.to_dict(include_variants=False),
            "model_variant": resolved_variant.to_dict(),
            "execution_maturity": {
                "model_family": resolved_family.maturity.value,
                "benchmark_task": resolved_task.maturity.value,
            },
        },
    )
    paths = ExperimentPaths.from_config(artifacts_root, config)
    return PlannedReleaseExperiment(
        release=release,
        task=resolved_task,
        model_family=resolved_family,
        model_variant=resolved_variant,
        config=config,
        paths=paths,
    )


def load_planned_release_experiment(path: str | Path) -> PlannedReleaseExperiment:
    resolved_path = Path(path).resolve()
    plan_path = resolved_path / "plan.json" if resolved_path.is_dir() else resolved_path
    if not plan_path.exists():
        raise FileNotFoundError(f"Planned experiment path does not exist: {plan_path}")

    run_dir = plan_path.parent
    config_payload = load_json_object(run_dir / "config.json")
    manifest_payload = load_json_object(run_dir / "manifest.json")
    plan_payload = load_json_object(plan_path)
    validate_run_manifest_payload(manifest_payload)

    config = ExperimentConfig.from_dict(config_payload)
    expected_paths = ExperimentPaths.from_config(run_dir.parent, config)
    if expected_paths.run_dir.resolve() != run_dir.resolve():
        raise ValueError(
            "Planned experiment run directory does not match the deterministic config digest."
        )
    if str(manifest_payload.get("run_id")) != expected_paths.run_id:
        raise ValueError(
            "Experiment manifest run_id does not match the deterministic "
            "config digest."
        )

    release_payload = coerce_dict(plan_payload.get("release"))
    model_family_payload = coerce_dict(plan_payload.get("model_family"))
    model_variant_payload = coerce_dict(plan_payload.get("model_variant"))
    task_payload = coerce_dict(plan_payload.get("task"))

    release = load_published_release(str(release_payload["bundle_dir"]))
    if config.dataset != release.release_id:
        raise ValueError(
            "Experiment config dataset does not match the published release referenced by the plan."
        )

    resolved_family = resolve_model_family(str(model_family_payload["name"]))
    resolved_task = resolve_benchmark_task(str(task_payload["name"]))
    resolved_task.validate_model_family(resolved_family)
    resolved_task.validate_release(release)

    resolved_variant = resolved_family.resolve_variant(str(model_variant_payload["name"]))
    if config.model != resolved_variant.name:
        raise ValueError(
            "Experiment config model does not match the planned model variant."
        )

    config_release_payload = coerce_dict(config.extras.get("release", {}))
    if (
        config_release_payload
        and str(config_release_payload.get("release_id", "")) != release.release_id
    ):
        raise ValueError(
            "Experiment config release extras do not match the planned "
            "published release."
        )

    return PlannedReleaseExperiment(
        release=release,
        task=resolved_task,
        model_family=resolved_family,
        model_variant=resolved_variant,
        config=config,
        paths=expected_paths,
    )
