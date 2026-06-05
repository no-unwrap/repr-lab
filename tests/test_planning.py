from __future__ import annotations

import json
from pathlib import Path

import pytest

from repr_lab.main import main
from repr_lab.planning import load_planned_release_experiment, plan_release_experiment


def test_plan_release_experiment_writes_deterministic_plan(
    tmp_path: Path, release_bundle: Path
) -> None:
    plan = plan_release_experiment(
        release_bundle,
        model_family="vjepa2",
        task="frozen-feature-localization-probe",
        artifacts_root=tmp_path / "artifacts",
    ).write()

    assert plan.model_variant.name == "vjepa2_1_vit_base_384"
    assert plan.paths.config_path.exists()
    assert plan.paths.manifest_path.exists()
    assert plan.paths.plan_path.exists()

    payload = json.loads(plan.paths.plan_path.read_text(encoding="utf-8"))
    assert payload["task"]["name"] == "frozen-feature-localization-probe"
    assert payload["model_family"]["name"] == "vjepa2"
    assert payload["model_variant"]["name"] == "vjepa2_1_vit_base_384"
    assert payload["execution_maturity"]["model_family"] == "runnable"
    assert payload["paths"]["benchmark_result_path"].endswith("benchmark_result.json")


def test_plan_release_experiment_cli_prints_written_plan(
    tmp_path: Path, release_bundle: Path, capsys
) -> None:
    exit_code = main(
        [
            "plan-release-experiment",
            "--bundle-dir",
            str(release_bundle),
            "--model",
            "sam1",
            "--task",
            "automatic-mask-proposal",
            "--artifacts-root",
            str(tmp_path / "artifacts"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["model_family"]["name"] == "segment-anything"
    assert payload["model_variant"]["name"] == "vit_b"
    assert Path(payload["paths"]["plan_path"]).exists()


def test_plan_release_experiment_rejects_incompatible_family(
    tmp_path: Path, release_bundle: Path
) -> None:
    with pytest.raises(ValueError, match="not compatible with model family"):
        plan_release_experiment(
            release_bundle,
            model_family="vjepa2",
            task="promptable-mask-refinement",
            artifacts_root=tmp_path / "artifacts",
        )


def test_load_planned_release_experiment_round_trips_written_plan(
    tmp_path: Path, release_bundle: Path
) -> None:
    planned = plan_release_experiment(
        release_bundle,
        model_family="vjepa2",
        task="frozen-feature-localization-probe",
        artifacts_root=tmp_path / "artifacts",
    ).write()

    loaded = load_planned_release_experiment(planned.paths.run_dir)

    assert loaded.paths.run_dir == planned.paths.run_dir
    assert loaded.release.release_id == planned.release.release_id
    assert loaded.task.name == planned.task.name
    assert loaded.model_variant.name == planned.model_variant.name
