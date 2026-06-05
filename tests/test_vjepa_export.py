from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from repr_lab import (
    VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_NAME,
    VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_VERSION,
    LocalFeatureExport,
    VJEPALocalRawMediaExport,
    optional_raw_media_runtime_contract,
)


def test_vjepa_local_raw_media_export_round_trips_explicit_contract(tmp_path: Path) -> None:
    export_path = tmp_path / "vjepa_export.npz"
    export = VJEPALocalRawMediaExport.create(
        layers={"backbone_last_hidden_state": np.asarray([[0.1, 0.2], [0.3, 0.4]])},
        sample_ids=np.asarray([1, 2]),
        model_variant="vjepa2_1_vit_base_384",
        metadata={"default_layer": "backbone_last_hidden_state"},
    )
    export.save(export_path)

    loaded = VJEPALocalRawMediaExport.load(export_path)
    payload = loaded.to_dict()

    assert payload["contract_name"] == VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_NAME
    assert payload["contract_version"] == VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_VERSION
    assert payload["compatibility_mode"] == "explicit_contract"
    assert payload["model_family"] == "vjepa2"
    assert payload["model_variant"] == "vjepa2_1_vit_base_384"
    assert payload["sample_count"] == 2


def test_vjepa_local_raw_media_export_accepts_legacy_local_feature_export(tmp_path: Path) -> None:
    export_path = tmp_path / "legacy_export.npz"
    LocalFeatureExport(
        layers={"backbone_last_hidden_state": np.asarray([[0.1, 0.2], [0.3, 0.4]])},
        sample_ids=np.asarray([1, 2]),
        sample_id_kind="asset_id",
        metadata={
            "default_layer": "backbone_last_hidden_state",
            "model_family": "vjepa2",
            "model_variant": "vjepa2_1_vit_base_384",
        },
    ).save(export_path)

    loaded = VJEPALocalRawMediaExport.load(export_path)

    assert loaded.compatibility_mode == "legacy_vjepa2_metadata"
    assert (
        loaded.feature_export.metadata["adapter_contract_name"]
        == VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_NAME
    )


def test_vjepa_local_raw_media_export_rejects_wrong_model_family(tmp_path: Path) -> None:
    export_path = tmp_path / "wrong_family.npz"
    LocalFeatureExport(
        layers={"backbone_last_hidden_state": np.asarray([[0.1, 0.2]])},
        sample_ids=np.asarray([1]),
        metadata={
            "default_layer": "backbone_last_hidden_state",
            "model_family": "sam3",
            "model_variant": "sam3",
        },
    ).save(export_path)

    with pytest.raises(ValueError, match="model_family"):
        VJEPALocalRawMediaExport.load(export_path)


def test_optional_raw_media_runtime_contract_is_explicit() -> None:
    contract = optional_raw_media_runtime_contract().to_dict()

    assert contract["contract_name"] == "repr_lab_optional_raw_media_runtime"
    assert contract["contract_version"] == "0.1.0"
    assert contract["packages"] == ["torch", "torchvision", "PIL", "transformers"]
