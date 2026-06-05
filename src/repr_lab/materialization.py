from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from repr_lab._parsing import load_json_object
from repr_lab.checkpointing import ResolvedCheckpoint, load_checkpoint_manifest
from repr_lab.features import FeatureCollection, _to_array_map
from repr_lab.planning import PlannedReleaseExperiment, load_planned_release_experiment

SUPPORTED_MATERIALIZATION_KEYS = {("vjepa2", "frozen-feature-localization-probe")}
SUPPORTED_SAMPLE_ID_KINDS = {"asset_id", "file_name"}
TRAIN_SPLIT_NAMES = {"train", "training"}
EVAL_SPLIT_NAMES = {"test", "testing", "val", "valid", "validation", "eval", "evaluation"}
VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_NAME = "repr_lab_vjepa_local_raw_media_export"
VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_VERSION = "0.1.0"
VJEPA_LOCAL_RAW_MEDIA_EXPORT_ADAPTER_NAME = "repr_lab.local_vjepa_export"
OPTIONAL_RAW_MEDIA_RUNTIME_CONTRACT_NAME = "repr_lab_optional_raw_media_runtime"
OPTIONAL_RAW_MEDIA_RUNTIME_CONTRACT_VERSION = "0.1.0"
OPTIONAL_RAW_MEDIA_RUNTIME_PACKAGES = ("torch", "torchvision", "PIL", "transformers")
LOCAL_REFERENCE_MODEL_FAMILIES = ("vjepa2",)
EXPLICIT_ADAPTER_COMPATIBILITY_MODE = "explicit_contract"
LEGACY_ADAPTER_COMPATIBILITY_MODE = "legacy_vjepa2_metadata"


@dataclass(frozen=True, slots=True)
class OptionalRawMediaRuntimeContract:
    contract_name: str = OPTIONAL_RAW_MEDIA_RUNTIME_CONTRACT_NAME
    contract_version: str = OPTIONAL_RAW_MEDIA_RUNTIME_CONTRACT_VERSION
    packages: tuple[str, ...] = OPTIONAL_RAW_MEDIA_RUNTIME_PACKAGES
    notes: tuple[str, ...] = (
        (
            "These packages remain optional because the bounded V-JEPA path "
            "consumes pre-exported local features rather than owning "
            "generalized raw-media execution."
        ),
        (
            "Broader raw-media automation stays out of scope until a later "
            "explicit adapter tranche lands."
        ),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_name": self.contract_name,
            "contract_version": self.contract_version,
            "packages": list(self.packages),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class LocalFeatureExport:
    layers: dict[str, np.ndarray]
    sample_ids: np.ndarray
    sample_id_kind: str = "asset_id"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_layers = _to_array_map(self.layers)
        sample_counts = {array.shape[0] for array in normalized_layers.values()}
        if len(sample_counts) != 1:
            raise ValueError("all layers must share the same number of samples")

        sample_ids = np.asarray(self.sample_ids)
        if sample_ids.ndim != 1:
            raise ValueError("sample_ids must be a 1D array")
        if sample_ids.shape[0] != next(iter(sample_counts)):
            raise ValueError("sample_ids must align with the feature sample count")
        if self.sample_id_kind not in SUPPORTED_SAMPLE_ID_KINDS:
            known = ", ".join(sorted(SUPPORTED_SAMPLE_ID_KINDS))
            raise ValueError(
                f"Unsupported sample_id_kind '{self.sample_id_kind}'. Known values: {known}"
            )

        object.__setattr__(self, "layers", normalized_layers)
        object.__setattr__(self, "sample_ids", sample_ids)

    @property
    def num_samples(self) -> int:
        return int(self.sample_ids.shape[0])

    def layer_names(self) -> list[str]:
        return sorted(self.layers)

    def default_layer(self) -> str:
        metadata_default = self.metadata.get("default_layer")
        if isinstance(metadata_default, str) and metadata_default in self.layers:
            return metadata_default
        return self.layer_names()[0]

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "__sample_ids__": self.sample_ids,
            "__sample_id_kind__": np.array(self.sample_id_kind),
            "__metadata__": np.array(json.dumps(self.metadata, sort_keys=True)),
            "__layer_names__": np.array(self.layer_names()),
        }
        for name, values in self.layers.items():
            payload[f"layer::{name}"] = values
        np.savez_compressed(output_path, **cast(Any, payload))

    @classmethod
    def load(cls, path: str | Path) -> "LocalFeatureExport":
        input_path = Path(path)
        with np.load(input_path, allow_pickle=False) as payload:
            layer_names = [str(name) for name in payload["__layer_names__"].tolist()]
            layers = {name: payload[f"layer::{name}"] for name in layer_names}
            sample_ids = payload["__sample_ids__"]
            sample_id_kind = str(payload["__sample_id_kind__"].item())
            metadata = json.loads(str(payload["__metadata__"].item()))
        return cls(
            layers=layers,
            sample_ids=sample_ids,
            sample_id_kind=sample_id_kind,
            metadata=metadata,
        )

    def sample_ids_as_strings(self) -> list[str]:
        return [str(sample_id) for sample_id in self.sample_ids.tolist()]


@dataclass(frozen=True, slots=True)
class VJEPALocalRawMediaExport:
    """Bounded local V-JEPA raw-media export adapter contract."""

    feature_export: LocalFeatureExport
    compatibility_mode: str
    runtime_contract: OptionalRawMediaRuntimeContract = field(
        default_factory=OptionalRawMediaRuntimeContract
    )

    @classmethod
    def create(
        cls,
        *,
        layers: dict[str, np.ndarray],
        sample_ids: np.ndarray,
        model_variant: str,
        sample_id_kind: str = "asset_id",
        metadata: dict[str, Any] | None = None,
    ) -> "VJEPALocalRawMediaExport":
        normalized_metadata = _normalized_vjepa_export_metadata(
            metadata,
            model_variant=model_variant,
            compatibility_mode=EXPLICIT_ADAPTER_COMPATIBILITY_MODE,
        )
        return cls(
            feature_export=LocalFeatureExport(
                layers=layers,
                sample_ids=sample_ids,
                sample_id_kind=sample_id_kind,
                metadata=normalized_metadata,
            ),
            compatibility_mode=EXPLICIT_ADAPTER_COMPATIBILITY_MODE,
        )

    @classmethod
    def load(cls, path: str | Path) -> "VJEPALocalRawMediaExport":
        export = LocalFeatureExport.load(path)
        metadata = export.metadata
        model_family = str(metadata.get("model_family", "")).strip()
        if model_family and model_family != "vjepa2":
            raise ValueError("V-JEPA raw-media export model_family must be 'vjepa2'.")
        model_variant = metadata.get("model_variant")
        if not isinstance(model_variant, str) or not model_variant.strip():
            raise ValueError("V-JEPA raw-media export metadata must include model_variant.")

        raw_contract_name = metadata.get("adapter_contract_name")
        raw_contract_version = metadata.get("adapter_contract_version")
        if raw_contract_name is None and raw_contract_version is None:
            compatibility_mode = LEGACY_ADAPTER_COMPATIBILITY_MODE
        else:
            contract_name = str(raw_contract_name)
            if contract_name != VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_NAME:
                raise ValueError("Unsupported V-JEPA adapter_contract_name.")
            contract_version = str(raw_contract_version)
            supported_major = VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_VERSION.split(".", 1)[0]
            if contract_version.split(".", 1)[0] != supported_major:
                raise ValueError("Unsupported V-JEPA adapter_contract_version major.")
            compatibility_mode = EXPLICIT_ADAPTER_COMPATIBILITY_MODE

        normalized_metadata = _normalized_vjepa_export_metadata(
            metadata,
            model_variant=model_variant,
            compatibility_mode=compatibility_mode,
        )
        normalized_export = LocalFeatureExport(
            layers=export.layers,
            sample_ids=export.sample_ids,
            sample_id_kind=export.sample_id_kind,
            metadata=normalized_metadata,
        )
        return cls(
            feature_export=normalized_export,
            compatibility_mode=compatibility_mode,
        )

    @property
    def model_variant(self) -> str:
        return str(self.feature_export.metadata["model_variant"])

    def save(self, path: str | Path) -> None:
        self.feature_export.save(path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_name": VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_NAME,
            "contract_version": VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_VERSION,
            "adapter_name": VJEPA_LOCAL_RAW_MEDIA_EXPORT_ADAPTER_NAME,
            "compatibility_mode": self.compatibility_mode,
            "model_family": str(self.feature_export.metadata["model_family"]),
            "model_variant": self.model_variant,
            "sample_id_kind": self.feature_export.sample_id_kind,
            "sample_count": self.feature_export.num_samples,
            "layer_names": self.feature_export.layer_names(),
            "default_layer": self.feature_export.default_layer(),
            "runtime_contract": self.runtime_contract.to_dict(),
            "metadata": dict(self.feature_export.metadata),
        }


def optional_raw_media_runtime_contract() -> OptionalRawMediaRuntimeContract:
    return OptionalRawMediaRuntimeContract()


def _normalized_vjepa_export_metadata(
    metadata: dict[str, Any] | None,
    *,
    model_variant: str,
    compatibility_mode: str,
) -> dict[str, Any]:
    normalized = dict(metadata or {})
    normalized["model_family"] = "vjepa2"
    normalized["model_variant"] = model_variant
    normalized["adapter_contract_name"] = VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_NAME
    normalized["adapter_contract_version"] = VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_VERSION
    normalized["adapter_name"] = VJEPA_LOCAL_RAW_MEDIA_EXPORT_ADAPTER_NAME
    normalized["adapter_compatibility_mode"] = compatibility_mode
    normalized["runtime_contract_name"] = OPTIONAL_RAW_MEDIA_RUNTIME_CONTRACT_NAME
    normalized["runtime_contract_version"] = OPTIONAL_RAW_MEDIA_RUNTIME_CONTRACT_VERSION
    return normalized


@dataclass(frozen=True, slots=True)
class MaterializedReleaseFeatures:
    run_id: str
    model_family: str
    model_variant: str
    adapter_contract_name: str
    adapter_contract_version: str
    adapter_compatibility_mode: str
    selected_default_layer: str
    feature_export_path: Path
    checkpoint_manifest_path: Path
    copied_checkpoint_manifest_path: Path
    train_feature_path: Path
    test_feature_path: Path
    report_path: Path
    sample_id_kind: str
    train_sample_count: int
    test_sample_count: int
    layer_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "0.1.0",
            "run_id": self.run_id,
            "model_family": self.model_family,
            "model_variant": self.model_variant,
            "adapter_contract": {
                "contract_name": self.adapter_contract_name,
                "contract_version": self.adapter_contract_version,
                "compatibility_mode": self.adapter_compatibility_mode,
            },
            "selected_default_layer": self.selected_default_layer,
            "feature_export_path": str(self.feature_export_path),
            "checkpoint_manifest_path": str(self.checkpoint_manifest_path),
            "copied_checkpoint_manifest_path": str(self.copied_checkpoint_manifest_path),
            "sample_id_kind": self.sample_id_kind,
            "layer_names": list(self.layer_names),
            "outputs": {
                "train": str(self.train_feature_path),
                "test": str(self.test_feature_path),
                "report": str(self.report_path),
            },
            "sample_counts": {
                "train": self.train_sample_count,
                "test": self.test_sample_count,
            },
        }

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class _ResolvedReleaseSample:
    sample_index: int
    canonical_split: str
    source_split: str
    source_category_id: int


def materialized_feature_collection_path(
    plan_or_paths: PlannedReleaseExperiment | Path,
    *,
    split: str,
) -> Path:
    if isinstance(plan_or_paths, PlannedReleaseExperiment):
        artifacts_dir = plan_or_paths.paths.artifacts_dir
    else:
        artifacts_dir = Path(plan_or_paths)
    return artifacts_dir / f"materialized_features_{split}.npz"


def materialize_release_features(
    plan_path_or_run_dir: str | Path,
    *,
    feature_export: str | Path,
    checkpoint_manifest: str | Path,
) -> MaterializedReleaseFeatures:
    plan = load_planned_release_experiment(plan_path_or_run_dir)
    materialization_key = (plan.model_family.name, plan.task.name)
    if materialization_key not in SUPPORTED_MATERIALIZATION_KEYS:
        supported = ", ".join(
            f"{family}:{task}" for family, task in sorted(SUPPORTED_MATERIALIZATION_KEYS)
        )
        raise RuntimeError(
            "No local materialization path is registered for "
            f"{plan.model_family.name}:{plan.task.name}. Supported paths: {supported}"
        )

    export_path = Path(feature_export).resolve()
    manifest_input_path = Path(checkpoint_manifest).resolve()
    raw_media_export = VJEPALocalRawMediaExport.load(export_path)
    export = raw_media_export.feature_export
    resolved_checkpoint = load_checkpoint_manifest(manifest_input_path)
    _validate_checkpoint_manifest(resolved_checkpoint, plan=plan)
    _validate_export_metadata(export, plan=plan)

    release_payload = load_json_object(Path(plan.release.artifact_paths["annotations_coco"]))
    category_name_by_id, category_order, image_index, file_name_index, label_sets = (
        _build_release_indices(release_payload)
    )

    resolved_samples = _resolve_export_samples(
        export,
        image_index=image_index,
        file_name_index=file_name_index,
        label_sets=label_sets,
    )
    ordered_source_category_ids = [
        category_id
        for category_id in category_order
        if any(sample.source_category_id == category_id for sample in resolved_samples)
    ]
    if not ordered_source_category_ids:
        raise ValueError("Resolved export does not map to any labeled release categories.")
    label_index_by_source = {
        category_id: index for index, category_id in enumerate(ordered_source_category_ids)
    }

    plan.paths.create()
    copied_checkpoint_manifest_path = plan.paths.checkpoints_dir / "resolved_checkpoint.json"
    resolved_checkpoint.write_json(copied_checkpoint_manifest_path)

    train_indices, train_labels, test_indices, test_labels = _split_materialized_samples(
        resolved_samples,
        label_index_by_source=label_index_by_source,
    )
    train_source_splits = sorted(
        {sample.source_split for sample in resolved_samples if sample.canonical_split == "train"}
    )
    test_source_splits = sorted(
        {sample.source_split for sample in resolved_samples if sample.canonical_split == "test"}
    )
    label_names = [
        category_name_by_id[category_id] for category_id in ordered_source_category_ids
    ]
    default_layer = export.default_layer()
    materialized_at = datetime.now(timezone.utc).isoformat()
    common_metadata = {
        "schema_version": "0.1.0",
        "release_id": plan.release.release_id,
        "release_version": plan.release.release_version,
        "model_family": plan.model_family.name,
        "model_variant": plan.model_variant.name,
        "default_layer": default_layer,
        "label_names": label_names,
        "label_map": {str(index): name for index, name in enumerate(label_names)},
        "source_category_ids": ordered_source_category_ids,
        "materialized_at": materialized_at,
        "materializer": {
            "name": VJEPA_LOCAL_RAW_MEDIA_EXPORT_ADAPTER_NAME,
            "adapter": {
                "contract_name": VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_NAME,
                "contract_version": VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_VERSION,
                "compatibility_mode": raw_media_export.compatibility_mode,
            },
            "runtime_contract": optional_raw_media_runtime_contract().to_dict(),
            "feature_export_path": str(export_path),
            "feature_export_metadata": export.metadata,
            "sample_id_kind": export.sample_id_kind,
            "checkpoint_manifest_path": str(copied_checkpoint_manifest_path),
            "checkpoint_manifest_input_path": str(manifest_input_path),
        },
        "checkpoint": resolved_checkpoint.to_feature_metadata(),
    }

    train_collection = FeatureCollection(
        layers=_subset_layers(export.layers, train_indices),
        labels=np.asarray(train_labels, dtype=np.int64),
        split="train",
        metadata={**common_metadata, "source_split_names": train_source_splits},
    )
    test_collection = FeatureCollection(
        layers=_subset_layers(export.layers, test_indices),
        labels=np.asarray(test_labels, dtype=np.int64),
        split="test",
        metadata={**common_metadata, "source_split_names": test_source_splits},
    )

    train_feature_path = materialized_feature_collection_path(plan, split="train")
    test_feature_path = materialized_feature_collection_path(plan, split="test")
    train_collection.save(train_feature_path)
    test_collection.save(test_feature_path)

    report = MaterializedReleaseFeatures(
        run_id=plan.paths.run_id,
        model_family=plan.model_family.name,
        model_variant=plan.model_variant.name,
        adapter_contract_name=VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_NAME,
        adapter_contract_version=VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_VERSION,
        adapter_compatibility_mode=raw_media_export.compatibility_mode,
        selected_default_layer=default_layer,
        feature_export_path=export_path,
        checkpoint_manifest_path=manifest_input_path,
        copied_checkpoint_manifest_path=copied_checkpoint_manifest_path,
        train_feature_path=train_feature_path,
        test_feature_path=test_feature_path,
        report_path=plan.paths.artifacts_dir / "feature_materialization.json",
        sample_id_kind=export.sample_id_kind,
        train_sample_count=train_collection.num_samples,
        test_sample_count=test_collection.num_samples,
        layer_names=tuple(export.layer_names()),
    )
    report.write_json(report.report_path)
    return report


def _validate_checkpoint_manifest(
    checkpoint: ResolvedCheckpoint,
    *,
    plan: PlannedReleaseExperiment,
) -> None:
    expected_spec = plan.model_variant.checkpoint
    if expected_spec is None:
        raise RuntimeError(
            f"Model variant '{plan.model_variant.name}' does not define checkpoint provenance."
        )
    if checkpoint.spec.source_kind is not expected_spec.source_kind:
        raise ValueError("Checkpoint manifest source_kind does not match the planned model.")
    if checkpoint.spec.locator != expected_spec.locator:
        raise ValueError("Checkpoint manifest locator does not match the planned model.")
    if checkpoint.model_family and checkpoint.model_family != plan.model_family.name:
        raise ValueError("Checkpoint manifest model_family does not match the planned model.")
    if checkpoint.model_variant and checkpoint.model_variant != plan.model_variant.name:
        raise ValueError("Checkpoint manifest model_variant does not match the planned model.")


def _validate_export_metadata(
    export: LocalFeatureExport,
    *,
    plan: PlannedReleaseExperiment,
) -> None:
    model_family = export.metadata.get("model_family")
    if model_family is not None and str(model_family) != plan.model_family.name:
        raise ValueError("Feature export model_family does not match the planned model.")
    model_variant = export.metadata.get("model_variant")
    if model_variant is not None and str(model_variant) != plan.model_variant.name:
        raise ValueError("Feature export model_variant does not match the planned model.")


def _build_release_indices(
    payload: dict[str, object],
) -> tuple[
    dict[int, str],
    list[int],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, set[int]],
]:
    images = _coerce_object_list(payload.get("images"), field_name="images")
    categories = _coerce_object_list(payload.get("categories"), field_name="categories")
    annotations = _coerce_object_list(payload.get("annotations"), field_name="annotations")

    category_name_by_id: dict[int, str] = {}
    category_order: list[int] = []
    for category in categories:
        category_id = _coerce_int(category.get("id"), field_name="categories[].id")
        category_name_by_id[category_id] = _require_non_empty_string(
            category.get("name"),
            field_name="categories[].name",
        )
        category_order.append(category_id)

    image_index: dict[str, dict[str, str]] = {}
    file_name_index: dict[str, dict[str, str]] = {}
    for image in images:
        asset_id = str(_coerce_int(image.get("id"), field_name="images[].id"))
        file_name = _require_non_empty_string(
            image.get("file_name"),
            field_name="images[].file_name",
        )
        source_split = str(image.get("split") or "unspecified")
        image_payload = {
            "asset_id": asset_id,
            "file_name": file_name,
            "source_split": source_split,
        }
        image_index[asset_id] = image_payload
        file_name_index[file_name] = image_payload

    label_sets: dict[str, set[int]] = defaultdict(set)
    for annotation in annotations:
        asset_id = str(_coerce_int(annotation.get("image_id"), field_name="annotations[].image_id"))
        category_id = _coerce_int(
            annotation.get("category_id"),
            field_name="annotations[].category_id",
        )
        if category_id not in category_name_by_id:
            raise ValueError(
                "annotations[].category_id references unknown category id "
                f"{category_id}"
            )
        label_sets[asset_id].add(category_id)

    return category_name_by_id, category_order, image_index, file_name_index, label_sets


def _resolve_export_samples(
    export: LocalFeatureExport,
    *,
    image_index: dict[str, dict[str, str]],
    file_name_index: dict[str, dict[str, str]],
    label_sets: dict[str, set[int]],
) -> list[_ResolvedReleaseSample]:
    resolved: list[_ResolvedReleaseSample] = []
    sample_ids = export.sample_ids_as_strings()
    for sample_index, sample_id in enumerate(sample_ids):
        if export.sample_id_kind == "asset_id":
            image_payload = image_index.get(sample_id)
        else:
            image_payload = file_name_index.get(sample_id)
        if image_payload is None:
            raise ValueError(
                f"Feature export sample '{sample_id}' does not map to a release asset."
            )

        asset_id = image_payload["asset_id"]
        source_split = image_payload["source_split"]
        canonical_split = _canonicalize_split(source_split)
        if canonical_split is None:
            raise ValueError(
                f"Release asset '{asset_id}' uses unsupported split '{source_split}'."
            )

        labels = label_sets.get(asset_id, set())
        if not labels:
            raise ValueError(f"Release asset '{asset_id}' does not have any labels.")
        if len(labels) != 1:
            raise ValueError(
                f"Release asset '{asset_id}' has multiple label classes; "
                "bounded materialization requires one class per asset."
            )
        resolved.append(
            _ResolvedReleaseSample(
                sample_index=sample_index,
                canonical_split=canonical_split,
                source_split=source_split,
                source_category_id=next(iter(labels)),
            )
        )
    return resolved


def _split_materialized_samples(
    samples: list[_ResolvedReleaseSample],
    *,
    label_index_by_source: dict[int, int],
) -> tuple[list[int], list[int], list[int], list[int]]:
    train_indices: list[int] = []
    train_labels: list[int] = []
    test_indices: list[int] = []
    test_labels: list[int] = []
    for sample in samples:
        label_index = label_index_by_source[sample.source_category_id]
        if sample.canonical_split == "train":
            train_indices.append(sample.sample_index)
            train_labels.append(label_index)
        else:
            test_indices.append(sample.sample_index)
            test_labels.append(label_index)
    if not train_indices:
        raise ValueError("Materialized feature export did not resolve any train samples.")
    if not test_indices:
        raise ValueError("Materialized feature export did not resolve any evaluation samples.")
    return train_indices, train_labels, test_indices, test_labels


def _subset_layers(
    layers: dict[str, np.ndarray],
    indices: list[int],
) -> dict[str, np.ndarray]:
    index_array = np.asarray(indices, dtype=np.int64)
    return {name: values[index_array] for name, values in layers.items()}


def _canonicalize_split(split_name: str) -> str | None:
    normalized = split_name.strip().lower()
    if normalized in TRAIN_SPLIT_NAMES:
        return "train"
    if normalized in EVAL_SPLIT_NAMES:
        return "test"
    return None


def _coerce_object_list(value: object, *, field_name: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    objects: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{field_name} must contain only objects")
        objects.append({str(key): inner_value for key, inner_value in item.items()})
    return objects


def _coerce_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    raise ValueError(f"{field_name} must be integer-compatible")


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value
