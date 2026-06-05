from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repr_lab.benchmark_result import BenchmarkResultRecord
from repr_lab.checkpointing import PreparedCheckpoint, ResolvedCheckpoint
from repr_lab.features import FeatureCollection
from repr_lab.planning import PlannedReleaseExperiment


@dataclass(frozen=True, slots=True)
class SummaryWarning:
    code: str
    detail: str
    severity: str = "warn"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class RunSummaryRecord:
    schema_version: str
    run_id: str
    status: str
    release_id: str
    release_version: str
    benchmark_task: str
    model_family: str
    model_variant: str
    selected_layer: str
    input_mode: str
    task_metrics: tuple[str, ...]
    task_outputs: tuple[str, ...]
    metrics: dict[str, float]
    release_counts: dict[str, int]
    release_split_summary: dict[str, int]
    sample_counts: dict[str, int]
    source_splits: dict[str, list[str]]
    label_names: list[str]
    feature_layers: tuple[str, ...]
    provenance: dict[str, Any]
    warnings: tuple[SummaryWarning, ...] = ()
    artifact_paths: dict[str, str] | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status,
            "release_id": self.release_id,
            "release_version": self.release_version,
            "benchmark_task": self.benchmark_task,
            "model_family": self.model_family,
            "model_variant": self.model_variant,
            "selected_layer": self.selected_layer,
            "input_mode": self.input_mode,
            "task_metrics": list(self.task_metrics),
            "task_outputs": list(self.task_outputs),
            "metrics": self.metrics,
            "release_counts": self.release_counts,
            "release_split_summary": self.release_split_summary,
            "sample_counts": self.sample_counts,
            "source_splits": self.source_splits,
            "label_names": self.label_names,
            "feature_layers": list(self.feature_layers),
            "provenance": self.provenance,
        }
        if self.warnings:
            payload["warnings"] = [warning.to_dict() for warning in self.warnings]
        if self.artifact_paths is not None:
            payload["artifact_paths"] = self.artifact_paths
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def write_markdown(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_run_report(self), encoding="utf-8")


def build_run_summary(
    plan: PlannedReleaseExperiment,
    *,
    record: BenchmarkResultRecord,
    selected_layer: str,
    input_mode: str,
    train_collection: FeatureCollection,
    test_collection: FeatureCollection,
    label_names: list[str],
    checkpoint: PreparedCheckpoint | ResolvedCheckpoint | dict[str, Any] | None,
    artifact_paths: dict[str, str],
    notes: tuple[str, ...] = (),
) -> RunSummaryRecord:
    warnings: list[SummaryWarning] = []
    source_splits = {
        "train": _coerce_string_list(train_collection.metadata.get("source_split_names")),
        "test": _coerce_string_list(test_collection.metadata.get("source_split_names")),
    }

    if input_mode == "canonical_materialized":
        warnings.append(
            SummaryWarning(
                code="local_only_reference_path",
                detail=(
                    "This run used repr-lab's local-only checkpoint-manifest and feature-"
                    "materialization path; it does not imply a generalized raw-media adapter."
                ),
            )
        )

    checkpoint_payload = _checkpoint_to_payload(checkpoint)
    if isinstance(checkpoint, ResolvedCheckpoint) and checkpoint.local_only:
        warnings.append(
            SummaryWarning(
                code="local_checkpoint_manifest",
                detail=(
                    "Checkpoint provenance is local-only and was resolved outside automated "
                    "shared checkpoint preparation."
                ),
            )
        )

    test_source_splits = source_splits["test"]
    if test_source_splits and test_source_splits != ["test"]:
        warnings.append(
            SummaryWarning(
                code="evaluation_split_alias",
                detail=(
                    "Evaluation artifacts were derived from release split(s) "
                    f"{', '.join(test_source_splits)} and normalized to canonical 'test'."
                ),
            )
        )

    sample_counts = {
        "train": int(train_collection.num_samples),
        "test": int(test_collection.num_samples),
    }
    if sample_counts["train"] < 10 or sample_counts["test"] < 10:
        warnings.append(
            SummaryWarning(
                code="small_sample_reference_run",
                detail=(
                    "This run has small split sizes and should be treated as a bounded "
                    "reference benchmark, not a stable leaderboard result."
                ),
            )
        )

    return RunSummaryRecord(
        schema_version="0.1.0",
        run_id=record.run_id,
        status=record.status.value,
        release_id=record.release_id,
        release_version=record.release_version,
        benchmark_task=record.benchmark_task,
        model_family=record.model_family,
        model_variant=record.model_variant,
        selected_layer=selected_layer,
        input_mode=input_mode,
        task_metrics=plan.task.metrics,
        task_outputs=plan.task.outputs,
        metrics=record.metrics or {},
        release_counts=plan.release.counts,
        release_split_summary=plan.release.split_summary,
        sample_counts=sample_counts,
        source_splits=source_splits,
        label_names=label_names,
        feature_layers=tuple(train_collection.layer_names()),
        provenance={
            "checkpoint": checkpoint_payload,
            "train_feature_metadata": train_collection.metadata,
            "test_feature_metadata": test_collection.metadata,
        },
        warnings=tuple(warnings),
        artifact_paths=artifact_paths,
        notes=notes,
    )


def render_run_report(summary: RunSummaryRecord) -> str:
    lines = [
        "# Run Report",
        "",
        f"- run id: `{summary.run_id}`",
        f"- status: `{summary.status}`",
        f"- task: `{summary.benchmark_task}`",
        f"- model: `{summary.model_family}` / `{summary.model_variant}`",
        f"- selected layer: `{summary.selected_layer}`",
        f"- input mode: `{summary.input_mode}`",
        "",
        "## Metrics",
        "",
    ]
    for metric_name, metric_value in sorted(summary.metrics.items()):
        lines.append(f"- {metric_name}: `{metric_value:.6f}`")

    lines.extend(
        [
            "",
            "## Data",
            "",
            f"- release: `{summary.release_id}` `{summary.release_version}`",
            f"- train samples: `{summary.sample_counts['train']}`",
            f"- test samples: `{summary.sample_counts['test']}`",
            f"- label names: {', '.join(summary.label_names)}",
            "",
            "## Provenance",
            "",
        ]
    )

    checkpoint = summary.provenance.get("checkpoint")
    if isinstance(checkpoint, dict):
        lines.append(f"- checkpoint source: `{checkpoint.get('source_kind', 'unknown')}`")
        lines.append(f"- checkpoint locator: `{checkpoint.get('locator', 'unknown')}`")
        if checkpoint.get("local_path"):
            lines.append(f"- checkpoint local path: `{checkpoint['local_path']}`")
    else:
        lines.append("- checkpoint: not recorded")

    lines.extend(
        [
            f"- feature layers: {', '.join(summary.feature_layers)}",
            f"- train source splits: {', '.join(summary.source_splits['train']) or 'n/a'}",
            f"- test source splits: {', '.join(summary.source_splits['test']) or 'n/a'}",
        ]
    )

    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in summary.warnings:
            lines.append(f"- `{warning.code}`: {warning.detail}")

    if summary.artifact_paths:
        lines.extend(["", "## Artifacts", ""])
        for name, path in sorted(summary.artifact_paths.items()):
            lines.append(f"- `{name}`: `{path}`")

    if summary.notes:
        lines.extend(["", "## Notes", ""])
        for note in summary.notes:
            lines.append(f"- {note}")

    lines.append("")
    return "\n".join(lines)


def _checkpoint_to_payload(
    checkpoint: PreparedCheckpoint | ResolvedCheckpoint | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    if isinstance(checkpoint, dict):
        return checkpoint
    return checkpoint.to_dict()


def _coerce_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
