from __future__ import annotations

import json
from pathlib import Path

import pytest

from repr_lab import BenchmarkRunStatus, FeatureCollection, execute_release_plan
from repr_lab.planning import plan_release_experiment


def _write_release_bundle(bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "annotations.coco.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "scene_001.jpg"},
                    {"id": 2, "file_name": "scene_002.jpg"},
                ],
                "categories": [
                    {"id": 0, "name": "background"},
                    {"id": 1, "name": "signal-object"},
                ],
                "annotations": [
                    {"id": 10, "image_id": 1, "category_id": 0, "bbox": [1, 2, 3, 4]},
                    {"id": 11, "image_id": 2, "category_id": 1, "bbox": [2, 3, 4, 5]},
                ],
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "assets.parquet").write_text("parquet-placeholder", encoding="utf-8")
    (bundle_dir / "categories.parquet").write_text("parquet-placeholder", encoding="utf-8")
    (bundle_dir / "release_manifest.json").write_text(
        json.dumps(
            {
                "contract_name": "published_artifact_bundle_contract",
                "contract_version": "1.0.0",
                "bundle_kind": "dataset_release_bundle",
                "release_id": "demo-release",
                "release_version": "1.0.0",
                "publisher_repo": "label-lab",
                "publisher_commit_sha": "abc123",
                "canonical_annotation_format": "COCO",
                "source_formats": ["COCO"],
                "task_types": ["bbox"],
                "artifact_paths": {
                    "annotations_coco": "annotations.coco.json",
                    "assets_table": "assets.parquet",
                    "categories_table": "categories.parquet",
                },
                "lineage": [],
                "split_summary": {"train": 4, "test": 2},
            }
        ),
        encoding="utf-8",
    )


def _write_feature_collection(
    path: Path,
    *,
    split: str,
    release_id: str = "demo-release",
    release_version: str = "1.0.0",
    model_family: str = "vjepa2",
    model_variant: str = "vjepa2_1_vit_base_384",
) -> None:
    metadata = {
        "release_id": release_id,
        "release_version": release_version,
        "model_family": model_family,
        "model_variant": model_variant,
        "label_names": ["background", "signal-object"],
        "checkpoint": {
            "source_kind": "torch_hub",
            "locator": "facebookresearch/vjepa2",
            "revision": "main",
            "local_note": "Resolved outside repr-lab for local-only probe execution.",
        },
    }
    if split == "train":
        layers = {
            "backbone_last_hidden_state": [
                [0.0, 0.2, 0.1],
                [0.2, 0.1, 0.0],
                [0.9, 1.0, 0.8],
                [1.1, 0.9, 1.0],
            ]
        }
        labels = [0, 0, 1, 1]
    else:
        layers = {
            "backbone_last_hidden_state": [
                [0.1, 0.0, 0.2],
                [1.0, 1.1, 0.9],
            ]
        }
        labels = [0, 1]
    FeatureCollection(
        layers=layers,
        labels=labels,
        split=split,
        metadata=metadata,
    ).save(path)


def test_execute_release_plan_writes_benchmark_artifacts(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_release_bundle(bundle_dir)
    plan = plan_release_experiment(
        bundle_dir,
        model_family="vjepa2",
        task="frozen-feature-localization-probe",
        artifacts_root=tmp_path / "artifacts",
    ).write()
    train_features = tmp_path / "train_features.npz"
    test_features = tmp_path / "test_features.npz"
    _write_feature_collection(train_features, split="train")
    _write_feature_collection(test_features, split="test")

    record = execute_release_plan(
        plan.paths.run_dir,
        train_features=train_features,
        test_features=test_features,
    )

    assert record.status is BenchmarkRunStatus.SUCCEEDED
    assert plan.paths.benchmark_result_path.exists()
    assert plan.paths.result_path.exists()
    assert Path(record.artifact_paths["probe_metrics"]).exists()
    assert Path(record.artifact_paths["feature_sources"]).exists()
    assert Path(record.artifact_paths["run_summary"]).exists()
    assert Path(record.artifact_paths["run_report"]).exists()

    benchmark_payload = json.loads(plan.paths.benchmark_result_path.read_text(encoding="utf-8"))
    assert benchmark_payload["status"] == "succeeded"
    assert benchmark_payload["benchmark_task"] == "frozen-feature-localization-probe"
    assert benchmark_payload["metrics"]["top1_accuracy"] == pytest.approx(1.0)

    feature_source_payload = json.loads(
        Path(record.artifact_paths["feature_sources"]).read_text(encoding="utf-8")
    )
    assert feature_source_payload["sources"]["train"]["metadata"]["checkpoint"]["locator"] == (
        "facebookresearch/vjepa2"
    )
    run_summary_payload = json.loads(
        Path(record.artifact_paths["run_summary"]).read_text(encoding="utf-8")
    )
    assert run_summary_payload["input_mode"] == "explicit_external"
    assert run_summary_payload["provenance"]["checkpoint"]["locator"] == "facebookresearch/vjepa2"
    run_report = Path(record.artifact_paths["run_report"]).read_text(encoding="utf-8")
    assert "# Run Report" in run_report


def test_execute_release_plan_rejects_feature_metadata_mismatch(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_release_bundle(bundle_dir)
    plan = plan_release_experiment(
        bundle_dir,
        model_family="vjepa2",
        task="frozen-feature-localization-probe",
        artifacts_root=tmp_path / "artifacts",
    ).write()
    train_features = tmp_path / "train_features.npz"
    test_features = tmp_path / "test_features.npz"
    _write_feature_collection(train_features, split="train", release_id="wrong-release")
    _write_feature_collection(test_features, split="test")

    with pytest.raises(ValueError, match="release_id"):
        execute_release_plan(
            plan.paths.plan_path,
            train_features=train_features,
            test_features=test_features,
        )


def test_execute_release_plan_rejects_split_mismatch(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_release_bundle(bundle_dir)
    plan = plan_release_experiment(
        bundle_dir,
        model_family="vjepa2",
        task="frozen-feature-localization-probe",
        artifacts_root=tmp_path / "artifacts",
    ).write()
    train_features = tmp_path / "train_features.npz"
    test_features = tmp_path / "test_features.npz"
    _write_feature_collection(train_features, split="test")
    _write_feature_collection(test_features, split="test")

    with pytest.raises(ValueError, match="Expected a 'train' feature collection"):
        execute_release_plan(
            plan.paths.run_dir,
            train_features=train_features,
            test_features=test_features,
        )


def test_execute_release_plan_rejects_asymmetric_feature_input(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_release_bundle(bundle_dir)
    plan = plan_release_experiment(
        bundle_dir,
        model_family="vjepa2",
        task="frozen-feature-localization-probe",
        artifacts_root=tmp_path / "artifacts",
    ).write()
    train_features = tmp_path / "train_features.npz"
    _write_feature_collection(train_features, split="train")

    with pytest.raises(ValueError, match="Provide both"):
        execute_release_plan(
            plan.paths.run_dir,
            train_features=train_features,
        )
