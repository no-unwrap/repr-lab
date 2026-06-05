from __future__ import annotations

import json
from pathlib import Path

import pytest

from repr_lab import (
    VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_NAME,
    BenchmarkRunStatus,
    FeatureCollection,
    VJEPALocalRawMediaExport,
    execute_release_plan,
    materialize_release_features,
    plan_release_experiment,
    record_local_checkpoint,
)


def _write_release_bundle(bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "annotations.coco.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "images/scene_001.jpg",
                        "width": 640,
                        "height": 480,
                        "split": "train",
                    },
                    {
                        "id": 2,
                        "file_name": "images/scene_002.jpg",
                        "width": 640,
                        "height": 480,
                        "split": "train",
                    },
                    {
                        "id": 3,
                        "file_name": "images/scene_003.jpg",
                        "width": 640,
                        "height": 480,
                        "split": "val",
                    },
                    {
                        "id": 4,
                        "file_name": "images/scene_004.jpg",
                        "width": 640,
                        "height": 480,
                        "split": "val",
                    },
                ],
                "categories": [
                    {"id": 10, "name": "signal-object"},
                    {"id": 20, "name": "reference-marker"},
                ],
                "annotations": [
                    {"id": 100, "image_id": 1, "category_id": 10, "bbox": [1, 2, 3, 4]},
                    {"id": 101, "image_id": 2, "category_id": 20, "bbox": [4, 3, 2, 1]},
                    {"id": 102, "image_id": 3, "category_id": 10, "bbox": [1, 2, 3, 4]},
                    {"id": 103, "image_id": 4, "category_id": 20, "bbox": [4, 3, 2, 1]},
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
                "split_summary": {"train": 2, "val": 2},
            }
        ),
        encoding="utf-8",
    )


def _write_feature_export(path: Path) -> None:
    VJEPALocalRawMediaExport.create(
        layers={
            "backbone_last_hidden_state": [
                [0.0, 0.1, 0.0],
                [1.0, 1.1, 1.0],
                [0.1, 0.0, 0.2],
                [0.9, 1.0, 1.1],
            ]
        },
        sample_ids=[1, 2, 3, 4],
        sample_id_kind="asset_id",
        model_variant="vjepa2_1_vit_base_384",
        metadata={"default_layer": "backbone_last_hidden_state"},
    ).save(path)


def test_materialize_release_features_writes_canonical_feature_collections(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_release_bundle(bundle_dir)
    plan = plan_release_experiment(
        bundle_dir,
        model_family="vjepa2",
        task="frozen-feature-localization-probe",
        artifacts_root=tmp_path / "artifacts",
    ).write()
    export_path = tmp_path / "vjepa_export.npz"
    _write_feature_export(export_path)

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.pt").write_text("placeholder", encoding="utf-8")
    checkpoint_manifest_path = tmp_path / "checkpoint_manifest.json"
    record_local_checkpoint(
        "vjepa2",
        local_path=checkpoint_dir,
        output_path=checkpoint_manifest_path,
        files=("model.pt",),
    )

    materialized = materialize_release_features(
        plan.paths.run_dir,
        feature_export=export_path,
        checkpoint_manifest=checkpoint_manifest_path,
    )

    train_collection = FeatureCollection.load(materialized.train_feature_path)
    test_collection = FeatureCollection.load(materialized.test_feature_path)

    assert materialized.copied_checkpoint_manifest_path.exists()
    assert materialized.adapter_contract_name == VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_NAME
    assert materialized.adapter_compatibility_mode == "explicit_contract"
    assert train_collection.split == "train"
    assert test_collection.split == "test"
    assert train_collection.labels.tolist() == [0, 1]
    assert test_collection.labels.tolist() == [0, 1]
    assert train_collection.metadata["release_id"] == "demo-release"
    assert train_collection.metadata["source_category_ids"] == [10, 20]
    assert (
        train_collection.metadata["materializer"]["adapter"]["contract_name"]
        == VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_NAME
    )
    assert (
        train_collection.metadata["materializer"]["adapter"]["compatibility_mode"]
        == "explicit_contract"
    )
    assert train_collection.metadata["checkpoint"]["locator"] == "facebookresearch/vjepa2"
    assert train_collection.metadata["checkpoint"]["local_only"] is True

    report_payload = json.loads(materialized.report_path.read_text(encoding="utf-8"))
    assert report_payload["adapter_contract"]["contract_name"] == (
        VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_NAME
    )
    assert report_payload["adapter_contract"]["compatibility_mode"] == "explicit_contract"


def test_execute_release_plan_uses_materialized_features_by_default(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_release_bundle(bundle_dir)
    plan = plan_release_experiment(
        bundle_dir,
        model_family="vjepa2",
        task="frozen-feature-localization-probe",
        artifacts_root=tmp_path / "artifacts",
    ).write()
    export_path = tmp_path / "vjepa_export.npz"
    _write_feature_export(export_path)

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.pt").write_text("placeholder", encoding="utf-8")
    checkpoint_manifest_path = tmp_path / "checkpoint_manifest.json"
    record_local_checkpoint(
        "vjepa2",
        local_path=checkpoint_dir,
        output_path=checkpoint_manifest_path,
        files=("model.pt",),
    )
    materialize_release_features(
        plan.paths.run_dir,
        feature_export=export_path,
        checkpoint_manifest=checkpoint_manifest_path,
    )

    record = execute_release_plan(plan.paths.run_dir)

    assert record.status is BenchmarkRunStatus.SUCCEEDED
    assert record.metrics is not None
    assert record.metrics["top1_accuracy"] == pytest.approx(1.0)
    assert "canonical run-local feature collections" in record.notes[1]
    assert record.checkpoint is not None

    run_summary_payload = json.loads(
        Path(record.artifact_paths["run_summary"]).read_text(encoding="utf-8")
    )
    assert run_summary_payload["input_mode"] == "canonical_materialized"
    assert run_summary_payload["warnings"][0]["code"] == "local_only_reference_path"
    assert run_summary_payload["warnings"][1]["code"] == "local_checkpoint_manifest"
    assert run_summary_payload["warnings"][2]["code"] == "evaluation_split_alias"
    assert run_summary_payload["warnings"][3]["code"] == "small_sample_reference_run"
    assert run_summary_payload["provenance"]["checkpoint"]["model_family"] == "vjepa2"

    run_report = Path(record.artifact_paths["run_report"]).read_text(encoding="utf-8")
    assert "evaluation_split_alias" in run_report


def test_materialize_release_features_rejects_checkpoint_mismatch(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_release_bundle(bundle_dir)
    plan = plan_release_experiment(
        bundle_dir,
        model_family="vjepa2",
        task="frozen-feature-localization-probe",
        artifacts_root=tmp_path / "artifacts",
    ).write()
    export_path = tmp_path / "vjepa_export.npz"
    _write_feature_export(export_path)

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "weights.safetensors").write_text("placeholder", encoding="utf-8")
    checkpoint_manifest_path = tmp_path / "checkpoint_manifest.json"
    record_local_checkpoint(
        "sam3",
        local_path=checkpoint_dir,
        output_path=checkpoint_manifest_path,
        files=("weights.safetensors",),
    )

    with pytest.raises(ValueError, match="source_kind"):
        materialize_release_features(
            plan.paths.run_dir,
            feature_export=export_path,
            checkpoint_manifest=checkpoint_manifest_path,
        )
