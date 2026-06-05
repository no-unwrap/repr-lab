from __future__ import annotations

from pathlib import Path

import pytest

from repr_lab import FeatureCollection, LocalFeatureExport, materialize_release_features
from repr_lab.checkpointing import record_local_checkpoint
from repr_lab.planning import plan_release_experiment


def _shared_release_fixture_dir() -> Path | None:
    candidate = (
        Path(__file__).resolve().parents[2]
        / "label-lab"
        / "tests"
        / "fixtures"
        / "published_release_bundle_v1"
    )
    return candidate if candidate.exists() else None


def test_label_lab_shared_release_fixture_materializes_locally(tmp_path: Path) -> None:
    fixture_dir = _shared_release_fixture_dir()
    if fixture_dir is None:
        pytest.skip("Shared label-lab fixture is unavailable in this workspace.")

    plan = plan_release_experiment(
        fixture_dir,
        model_family="vjepa2",
        task="frozen-feature-localization-probe",
        artifacts_root=tmp_path / "artifacts",
    ).write()
    export_path = tmp_path / "shared_fixture_export.npz"
    LocalFeatureExport(
        layers={
            "backbone_last_hidden_state": [
                [0.0, 0.2, 0.1],
                [1.0, 0.9, 1.1],
            ]
        },
        sample_ids=[1, 2],
        sample_id_kind="asset_id",
        metadata={
            "model_family": "vjepa2",
            "model_variant": "vjepa2_1_vit_base_384",
            "default_layer": "backbone_last_hidden_state",
        },
    ).save(export_path)

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

    assert train_collection.num_samples == 1
    assert test_collection.num_samples == 1
    assert materialized.adapter_compatibility_mode == "legacy_vjepa2_metadata"
    assert train_collection.metadata["release_id"] == "shared-release-fixture-1-2-0"
    assert train_collection.metadata["source_split_names"] == ["train"]
    assert test_collection.metadata["source_split_names"] == ["val"]
    assert train_collection.metadata["source_category_ids"] == [10, 20]
