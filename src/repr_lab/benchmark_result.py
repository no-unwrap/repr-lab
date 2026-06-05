from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from repr_lab.checkpointing import PreparedCheckpoint, ResolvedCheckpoint
from repr_lab.planning import PlannedReleaseExperiment
from repr_lab.run_contract import contract_fields


class BenchmarkRunStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class BenchmarkResultRecord:
    contract_name: str
    contract_version: str
    run_directory_kind: str
    producer_repo: str
    schema_version: str
    run_id: str
    release_id: str
    release_version: str
    benchmark_task: str
    model_family: str
    model_variant: str
    status: BenchmarkRunStatus
    started_at: str
    finished_at: str | None = None
    metrics: dict[str, float] | None = None
    artifact_paths: dict[str, str] | None = None
    checkpoint: PreparedCheckpoint | ResolvedCheckpoint | None = None
    notes: tuple[str, ...] = ()

    @classmethod
    def from_plan(
        cls,
        plan: PlannedReleaseExperiment,
        *,
        status: BenchmarkRunStatus = BenchmarkRunStatus.PLANNED,
        started_at: str | None = None,
        finished_at: str | None = None,
        metrics: dict[str, float] | None = None,
        artifact_paths: dict[str, str] | None = None,
        checkpoint: PreparedCheckpoint | ResolvedCheckpoint | None = None,
        notes: tuple[str, ...] = (),
    ) -> "BenchmarkResultRecord":
        return cls(
            contract_name=contract_fields()["contract_name"],
            contract_version=contract_fields()["contract_version"],
            run_directory_kind=contract_fields()["run_directory_kind"],
            producer_repo=contract_fields()["producer_repo"],
            schema_version="0.1.0",
            run_id=plan.paths.run_id,
            release_id=plan.release.release_id,
            release_version=plan.release.release_version,
            benchmark_task=plan.task.name,
            model_family=plan.model_family.name,
            model_variant=plan.model_variant.name,
            status=status,
            started_at=started_at or datetime.now(timezone.utc).isoformat(),
            finished_at=finished_at,
            metrics=metrics,
            artifact_paths=artifact_paths,
            checkpoint=checkpoint,
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract_name": self.contract_name,
            "contract_version": self.contract_version,
            "run_directory_kind": self.run_directory_kind,
            "producer_repo": self.producer_repo,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "release_id": self.release_id,
            "release_version": self.release_version,
            "benchmark_task": self.benchmark_task,
            "model_family": self.model_family,
            "model_variant": self.model_variant,
            "status": self.status.value,
            "started_at": self.started_at,
        }
        if self.finished_at is not None:
            payload["finished_at"] = self.finished_at
        if self.metrics is not None:
            payload["metrics"] = self.metrics
        if self.artifact_paths is not None:
            payload["artifact_paths"] = self.artifact_paths
        if self.checkpoint is not None:
            payload["checkpoint"] = self.checkpoint.to_dict()
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
