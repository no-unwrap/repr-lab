from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from repr_lab.config import ExperimentConfig
from repr_lab.run_contract import contract_fields


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """Stable experiment metadata written alongside a run directory."""

    contract_name: str
    contract_version: str
    run_directory_kind: str
    producer_repo: str
    name: str
    dataset: str
    model: str
    seed: int
    run_id: str
    created_at: str
    tags: tuple[str, ...] = ()
    extras: dict[str, Any] | None = None

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        *,
        run_id: str | None = None,
        created_at: str | None = None,
    ) -> "ExperimentManifest":
        return cls(
            contract_name=contract_fields()["contract_name"],
            contract_version=contract_fields()["contract_version"],
            run_directory_kind=contract_fields()["run_directory_kind"],
            producer_repo=contract_fields()["producer_repo"],
            name=config.name,
            dataset=config.dataset,
            model=config.model,
            seed=config.seed,
            run_id=run_id or config.digest(),
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            tags=config.tags,
            extras=config.extras or None,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "contract_name": self.contract_name,
            "contract_version": self.contract_version,
            "run_directory_kind": self.run_directory_kind,
            "producer_repo": self.producer_repo,
            "name": self.name,
            "dataset": self.dataset,
            "model": self.model,
            "seed": self.seed,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "tags": list(self.tags),
        }
        if self.extras:
            payload["extras"] = self.extras
        return payload

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class ExperimentPaths:
    """Standardized on-disk layout for experiment artifacts."""

    root: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_id

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def analysis_dir(self) -> Path:
        return self.run_dir / "analysis"

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def result_path(self) -> Path:
        return self.run_dir / "result.json"

    @property
    def benchmark_result_path(self) -> Path:
        return self.run_dir / "benchmark_result.json"

    @property
    def plan_path(self) -> Path:
        return self.run_dir / "plan.json"

    @classmethod
    def from_config(cls, root: str | Path, config: ExperimentConfig) -> "ExperimentPaths":
        return cls(root=Path(root), run_id=config.digest())

    def create(self) -> "ExperimentPaths":
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        return self

    def feature_store_path(self, *, split: str = "train", name: str = "features") -> Path:
        return self.artifacts_dir / f"{name}_{split}.npz"

    def analysis_report_path(self, name: str = "layerwise_metrics") -> Path:
        return self.analysis_dir / f"{name}.json"

    def initialize(
        self,
        config: ExperimentConfig,
        *,
        created_at: str | None = None,
    ) -> "ExperimentPaths":
        self.create()
        config.write_json(self.config_path)
        ExperimentManifest.from_config(
            config,
            run_id=self.run_id,
            created_at=created_at,
        ).write_json(self.manifest_path)
        return self
