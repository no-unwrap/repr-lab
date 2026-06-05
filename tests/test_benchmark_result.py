from __future__ import annotations

import json
from pathlib import Path

from repr_lab import BenchmarkResultRecord, BenchmarkRunStatus
from repr_lab.planning import plan_release_experiment


def test_benchmark_result_record_writes_standard_json(
    tmp_path: Path, release_bundle: Path
) -> None:
    plan = plan_release_experiment(
        release_bundle,
        model_family="vjepa2",
        task="frozen-feature-localization-probe",
        artifacts_root=tmp_path / "artifacts",
    ).write()

    record = BenchmarkResultRecord.from_plan(
        plan,
        status=BenchmarkRunStatus.BLOCKED,
        metrics={"top1_accuracy": 0.0},
        artifact_paths={"plan": str(plan.paths.plan_path)},
        notes=("Checkpoint resolution not yet implemented for torch hub backends.",),
    )
    record.write_json(plan.paths.benchmark_result_path)

    payload = json.loads(plan.paths.benchmark_result_path.read_text(encoding="utf-8"))
    assert payload["contract_name"] == "repr_lab_run_directory_contract"
    assert payload["contract_version"] == "1.0.0"
    assert payload["run_directory_kind"] == "benchmark_run_directory"
    assert payload["producer_repo"] == "repr-lab"
    assert payload["status"] == "blocked"
    assert payload["benchmark_task"] == "frozen-feature-localization-probe"
    assert payload["model_family"] == "vjepa2"
    assert payload["artifact_paths"]["plan"] == str(plan.paths.plan_path)
