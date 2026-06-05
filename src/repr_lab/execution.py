from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from repr_lab.analysis import (
    AmbientDimensionStat,
    ClassGeometryStat,
    EffectiveRankStat,
    LayerwiseAnalysisResult,
    LayerwiseAnalyzer,
    ParticipationRatioStat,
)
from repr_lab.benchmark_result import BenchmarkResultRecord, BenchmarkRunStatus
from repr_lab.checkpointing import PreparedCheckpoint, ResolvedCheckpoint, load_checkpoint_manifest
from repr_lab.features import FeatureCollection
from repr_lab.materialization import materialized_feature_collection_path
from repr_lab.planning import (
    PlannedReleaseExperiment,
    load_planned_release_experiment,
)
from repr_lab.reporting import build_run_summary

SUPPORTED_EXECUTOR_KEYS = {("vjepa2", "frozen-feature-localization-probe")}
REQUIRED_FEATURE_METADATA_KEYS = (
    "release_id",
    "release_version",
    "model_family",
    "model_variant",
)


@dataclass(frozen=True, slots=True)
class LinearProbeModel:
    classes: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    ridge_alpha: float
    selected_layer: str

    def score(self, features: np.ndarray) -> np.ndarray:
        normalized = _normalize_features(features, mean=self.mean, std=self.std)
        return cast(np.ndarray, _augment_bias(normalized) @ self.weights)

    def predict(self, features: np.ndarray) -> np.ndarray:
        scores = self.score(features)
        return cast(np.ndarray, self.classes[np.argmax(scores, axis=1)])

    def write_npz(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            classes=self.classes,
            mean=self.mean,
            std=self.std,
            weights=self.weights,
            ridge_alpha=np.array(self.ridge_alpha, dtype=np.float64),
            selected_layer=np.array(self.selected_layer),
        )


def execute_release_plan(
    plan_path_or_run_dir: str | Path,
    *,
    train_features: str | Path | None = None,
    test_features: str | Path | None = None,
    layer: str | None = None,
    ridge_alpha: float = 1.0,
) -> BenchmarkResultRecord:
    plan = load_planned_release_experiment(plan_path_or_run_dir)
    executor_key = (plan.model_family.name, plan.task.name)
    if executor_key not in SUPPORTED_EXECUTOR_KEYS:
        supported = ", ".join(
            f"{family}:{task}" for family, task in sorted(SUPPORTED_EXECUTOR_KEYS)
        )
        raise RuntimeError(
            "No runnable executor is registered for "
            f"{plan.model_family.name}:{plan.task.name}. Supported paths: {supported}"
        )

    (
        resolved_train_features,
        resolved_test_features,
        feature_input_note,
        input_mode,
    ) = _resolve_feature_inputs(
        plan,
        train_features=train_features,
        test_features=test_features,
    )
    return _execute_vjepa2_frozen_feature_probe(
        plan,
        train_features=resolved_train_features,
        test_features=resolved_test_features,
        layer=layer,
        ridge_alpha=ridge_alpha,
        feature_input_note=feature_input_note,
        input_mode=input_mode,
    )


def _execute_vjepa2_frozen_feature_probe(
    plan: PlannedReleaseExperiment,
    *,
    train_features: str | Path,
    test_features: str | Path,
    layer: str | None,
    ridge_alpha: float,
    feature_input_note: str,
    input_mode: str,
) -> BenchmarkResultRecord:
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")

    started_at = datetime.now(timezone.utc).isoformat()
    train_source_path = Path(train_features).resolve()
    test_source_path = Path(test_features).resolve()
    train_collection = FeatureCollection.load(train_source_path)
    test_collection = FeatureCollection.load(test_source_path)

    _validate_feature_collection(train_collection, expected_split="train", plan=plan)
    _validate_feature_collection(test_collection, expected_split="test", plan=plan)
    selected_layer = _resolve_selected_layer(train_collection, test_collection, requested=layer)

    train_labels = _coerce_labels(train_collection.labels, split="train")
    test_labels = _coerce_labels(test_collection.labels, split="test")
    probe_model = _fit_linear_probe(
        train_collection.flatten_layer(selected_layer),
        train_labels,
        ridge_alpha=ridge_alpha,
        selected_layer=selected_layer,
    )
    scores = probe_model.score(test_collection.flatten_layer(selected_layer))
    predicted_labels = probe_model.predict(test_collection.flatten_layer(selected_layer))

    metrics = {
        "top1_accuracy": _top1_accuracy(test_labels, predicted_labels),
        "macro_f1": _macro_f1(test_labels, predicted_labels),
        "mean_average_precision": _mean_average_precision(
            test_labels,
            scores,
            classes=probe_model.classes,
        ),
    }

    artifacts_dir = plan.paths.artifacts_dir
    analysis_dir = plan.paths.analysis_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    train_feature_store_path = artifacts_dir / "frozen_features_train.npz"
    test_feature_store_path = artifacts_dir / "frozen_features_test.npz"
    probe_model_path = artifacts_dir / "linear_probe.npz"
    probe_metrics_path = artifacts_dir / "probe_metrics.json"
    feature_sources_path = artifacts_dir / "feature_sources.json"
    confusion_summary_path = analysis_dir / "confusion_summary.json"
    train_analysis_path = analysis_dir / "layerwise_metrics_train.json"
    test_analysis_path = analysis_dir / "layerwise_metrics_test.json"
    run_summary_path = artifacts_dir / "run_summary.json"
    run_report_path = analysis_dir / "run_report.md"

    train_collection.save(train_feature_store_path)
    test_collection.save(test_feature_store_path)
    probe_model.write_npz(probe_model_path)
    _build_layerwise_analysis(train_collection).write_json(train_analysis_path)
    _build_layerwise_analysis(test_collection).write_json(test_analysis_path)

    class_labels = _resolve_class_labels(
        probe_model.classes,
        train_collection=train_collection,
        test_collection=test_collection,
    )
    confusion_summary = _build_confusion_summary(
        true_labels=test_labels,
        predicted_labels=predicted_labels,
        classes=probe_model.classes,
        class_labels=class_labels,
        selected_layer=selected_layer,
    )
    _write_json(confusion_summary_path, confusion_summary)

    feature_sources = {
        "schema_version": "0.1.0",
        "run_id": plan.paths.run_id,
        "selected_layer": selected_layer,
        "sources": {
            "train": _build_feature_source_payload(
                input_path=train_source_path,
                copied_path=train_feature_store_path,
                collection=train_collection,
            ),
            "test": _build_feature_source_payload(
                input_path=test_source_path,
                copied_path=test_feature_store_path,
                collection=test_collection,
            ),
        },
    }
    _write_json(feature_sources_path, feature_sources)

    probe_metrics = {
        "schema_version": "0.1.0",
        "run_id": plan.paths.run_id,
        "benchmark_task": plan.task.name,
        "model_family": plan.model_family.name,
        "model_variant": plan.model_variant.name,
        "release_id": plan.release.release_id,
        "release_version": plan.release.release_version,
        "selected_layer": selected_layer,
        "ridge_alpha": ridge_alpha,
        "train_sample_count": int(train_collection.num_samples),
        "test_sample_count": int(test_collection.num_samples),
        "class_ids": [int(class_id) for class_id in probe_model.classes.tolist()],
        "class_labels": class_labels,
        "metrics": metrics,
        "feature_sources_path": str(feature_sources_path),
        "probe_model_path": str(probe_model_path),
    }
    _write_json(probe_metrics_path, probe_metrics)
    _write_json(
        plan.paths.result_path,
        {
            "run_id": plan.paths.run_id,
            "selected_layer": selected_layer,
            "metrics": metrics,
            "probe_metrics_path": str(probe_metrics_path),
        },
    )

    finished_at = datetime.now(timezone.utc).isoformat()
    artifact_paths = {
        "plan": str(plan.paths.plan_path),
        "train_feature_store": str(train_feature_store_path),
        "test_feature_store": str(test_feature_store_path),
        "probe_model": str(probe_model_path),
        "probe_metrics": str(probe_metrics_path),
        "confusion_summary": str(confusion_summary_path),
        "feature_sources": str(feature_sources_path),
        "train_analysis": str(train_analysis_path),
        "test_analysis": str(test_analysis_path),
        "run_summary": str(run_summary_path),
        "run_report": str(run_report_path),
    }
    execution_notes = (
        "Executed from an existing planned run directory; config.json and "
        "manifest.json were treated as canonical input.",
        feature_input_note,
        "Direct upstream V-JEPA checkpoint loading remains intentionally "
        "out of scope for this tranche.",
    )
    checkpoint = _resolve_checkpoint_provenance(train_collection, test_collection)
    record = BenchmarkResultRecord.from_plan(
        plan,
        status=BenchmarkRunStatus.SUCCEEDED,
        started_at=started_at,
        finished_at=finished_at,
        metrics=metrics,
        artifact_paths=artifact_paths,
        checkpoint=checkpoint
        if isinstance(checkpoint, (PreparedCheckpoint, ResolvedCheckpoint))
        else None,
        notes=execution_notes,
    )
    record.write_json(plan.paths.benchmark_result_path)
    summary = build_run_summary(
        plan,
        record=record,
        selected_layer=selected_layer,
        input_mode=input_mode,
        train_collection=train_collection,
        test_collection=test_collection,
        label_names=class_labels,
        checkpoint=checkpoint,
        artifact_paths=artifact_paths,
        notes=execution_notes,
    )
    summary.write_json(run_summary_path)
    summary.write_markdown(run_report_path)
    return record


def _resolve_feature_inputs(
    plan: PlannedReleaseExperiment,
    *,
    train_features: str | Path | None,
    test_features: str | Path | None,
) -> tuple[Path, Path, str, str]:
    if train_features is None and test_features is None:
        train_path = materialized_feature_collection_path(plan, split="train")
        test_path = materialized_feature_collection_path(plan, split="test")
        missing = [path.name for path in (train_path, test_path) if not path.exists()]
        if missing:
            joined = ", ".join(sorted(missing))
            raise FileNotFoundError(
                "No explicit feature inputs were provided, and the planned run "
                f"directory is missing canonical materialized features: {joined}"
            )
        return (
            train_path.resolve(),
            test_path.resolve(),
            "Consumed canonical run-local feature collections materialized into "
            "the planned run directory before execution.",
            "canonical_materialized",
        )

    if train_features is None or test_features is None:
        raise ValueError("Provide both train_features and test_features, or neither.")

    return (
        Path(train_features).resolve(),
        Path(test_features).resolve(),
        "Consumed explicit local-only frozen feature collections and copied "
        "them into the planned run directory before execution.",
        "explicit_external",
    )


def _resolve_checkpoint_provenance(
    train_collection: FeatureCollection,
    test_collection: FeatureCollection,
) -> PreparedCheckpoint | ResolvedCheckpoint | dict[str, Any] | None:
    metadata_candidates = [train_collection.metadata, test_collection.metadata]
    for metadata in metadata_candidates:
        materializer = metadata.get("materializer")
        if isinstance(materializer, dict):
            manifest_path = materializer.get("checkpoint_manifest_path")
            if isinstance(manifest_path, str) and manifest_path:
                path = Path(manifest_path)
                if path.exists():
                    return load_checkpoint_manifest(path)
        checkpoint = metadata.get("checkpoint")
        if isinstance(checkpoint, dict):
            return {str(key): value for key, value in checkpoint.items()}
    return None


def _validate_feature_collection(
    collection: FeatureCollection,
    *,
    expected_split: str,
    plan: PlannedReleaseExperiment,
) -> None:
    if collection.split != expected_split:
        raise ValueError(
            f"Expected a '{expected_split}' feature collection, got '{collection.split}'."
        )
    if collection.labels is None:
        raise ValueError(f"Feature collection for split '{expected_split}' must include labels.")

    missing = [key for key in REQUIRED_FEATURE_METADATA_KEYS if key not in collection.metadata]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(
            f"Feature collection for split '{expected_split}' is missing "
            f"required metadata: {joined}"
        )

    if str(collection.metadata["release_id"]) != plan.release.release_id:
        raise ValueError("Feature collection release_id does not match the planned release.")
    if str(collection.metadata["release_version"]) != plan.release.release_version:
        raise ValueError("Feature collection release_version does not match the planned release.")
    if str(collection.metadata["model_family"]) != plan.model_family.name:
        raise ValueError("Feature collection model_family does not match the planned model family.")
    if str(collection.metadata["model_variant"]) != plan.model_variant.name:
        raise ValueError(
            "Feature collection model_variant does not match the planned model variant."
        )


def _resolve_selected_layer(
    train_collection: FeatureCollection,
    test_collection: FeatureCollection,
    *,
    requested: str | None,
) -> str:
    common_layers = sorted(set(train_collection.layer_names()) & set(test_collection.layer_names()))
    if not common_layers:
        raise ValueError("Train and test feature collections do not share any feature layers.")

    if requested is not None:
        if requested not in common_layers:
            known = ", ".join(common_layers)
            raise KeyError(f"Unknown layer '{requested}'. Shared layers: {known}")
        return requested

    if len(common_layers) == 1:
        return common_layers[0]

    default_layer = train_collection.metadata.get("default_layer") or test_collection.metadata.get(
        "default_layer"
    )
    if isinstance(default_layer, str) and default_layer in common_layers:
        return default_layer

    known = ", ".join(common_layers)
    raise ValueError(
        "Multiple shared feature layers are available. Pass --layer explicitly or set "
        f"metadata.default_layer. Shared layers: {known}"
    )


def _coerce_labels(labels: np.ndarray | None, *, split: str) -> np.ndarray:
    if labels is None:
        raise ValueError(f"Feature collection for split '{split}' must include labels.")
    as_array = np.asarray(labels)
    if as_array.ndim != 1:
        raise ValueError(f"Labels for split '{split}' must be 1D.")
    coerced = as_array.astype(np.int64)
    if not np.array_equal(as_array, coerced):
        raise ValueError(f"Labels for split '{split}' must be integer-compatible.")
    return coerced


def _fit_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    ridge_alpha: float,
    selected_layer: str,
) -> LinearProbeModel:
    classes = np.unique(train_labels)
    if classes.size < 2:
        raise ValueError("Frozen-feature probe training requires at least two label classes.")

    class_to_index = {int(class_id): index for index, class_id in enumerate(classes.tolist())}
    normalized = _normalize_features(train_features)
    mean = normalized.mean(axis=0, keepdims=True)
    centered = normalized - mean
    std = centered.std(axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    standardized = centered / std
    design = _augment_bias(standardized)

    targets = np.zeros((design.shape[0], classes.size), dtype=np.float64)
    target_indices = np.array(
        [class_to_index[int(label)] for label in train_labels],
        dtype=np.int64,
    )
    targets[np.arange(design.shape[0]), target_indices] = 1.0

    regularization = np.eye(design.shape[1], dtype=np.float64)
    regularization[-1, -1] = 0.0
    weights = np.linalg.pinv(design.T @ design + ridge_alpha * regularization) @ design.T @ targets
    return LinearProbeModel(
        classes=classes.astype(np.int64, copy=False),
        mean=mean,
        std=std,
        weights=weights,
        ridge_alpha=ridge_alpha,
        selected_layer=selected_layer,
    )


def _normalize_features(
    features: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Frozen-feature probe executor expects a 2D feature matrix.")
    if mean is None:
        mean = np.zeros((1, matrix.shape[1]), dtype=np.float64)
    if std is None:
        std = np.ones((1, matrix.shape[1]), dtype=np.float64)
    return cast(np.ndarray, (matrix - mean) / std)


def _augment_bias(features: np.ndarray) -> np.ndarray:
    bias = np.ones((features.shape[0], 1), dtype=np.float64)
    return np.concatenate([features, bias], axis=1)


def _top1_accuracy(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    return float(np.mean(true_labels == predicted_labels))


def _macro_f1(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    class_ids = np.unique(np.concatenate([true_labels, predicted_labels]))
    f1_scores: list[float] = []
    for class_id in class_ids.tolist():
        truth = true_labels == class_id
        predicted = predicted_labels == class_id
        true_positive = int(np.sum(truth & predicted))
        false_positive = int(np.sum(~truth & predicted))
        false_negative = int(np.sum(truth & ~predicted))
        denominator = (2 * true_positive) + false_positive + false_negative
        f1_scores.append((2 * true_positive) / denominator if denominator else 0.0)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def _mean_average_precision(
    true_labels: np.ndarray,
    scores: np.ndarray,
    *,
    classes: np.ndarray,
) -> float:
    average_precisions: list[float] = []
    for class_index, class_id in enumerate(classes.tolist()):
        one_vs_rest = (true_labels == class_id).astype(np.int64)
        positives = int(np.sum(one_vs_rest))
        if positives == 0:
            continue
        ranked = np.argsort(-scores[:, class_index], kind="mergesort")
        ranked_truth = one_vs_rest[ranked]
        true_positives = np.cumsum(ranked_truth)
        precision = true_positives / (np.arange(ranked_truth.shape[0]) + 1)
        average_precisions.append(float(np.sum(precision[ranked_truth == 1]) / positives))
    return float(np.mean(average_precisions)) if average_precisions else 0.0


def _build_confusion_summary(
    *,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    classes: np.ndarray,
    class_labels: list[str],
    selected_layer: str,
) -> dict[str, Any]:
    class_to_index = {int(class_id): index for index, class_id in enumerate(classes.tolist())}
    confusion = np.zeros((classes.size, classes.size), dtype=np.int64)
    for truth, predicted in zip(true_labels.tolist(), predicted_labels.tolist(), strict=True):
        confusion[class_to_index[int(truth)], class_to_index[int(predicted)]] += 1
    return {
        "schema_version": "0.1.0",
        "selected_layer": selected_layer,
        "class_ids": [int(class_id) for class_id in classes.tolist()],
        "class_labels": class_labels,
        "matrix": confusion.tolist(),
    }


def _build_layerwise_analysis(collection: FeatureCollection) -> LayerwiseAnalysisResult:
    analyzer = LayerwiseAnalyzer(
        [
            AmbientDimensionStat(),
            EffectiveRankStat(),
            ParticipationRatioStat(),
            ClassGeometryStat(),
        ]
    )
    return analyzer.analyze(collection)


def _build_feature_source_payload(
    *,
    input_path: Path,
    copied_path: Path,
    collection: FeatureCollection,
) -> dict[str, Any]:
    return {
        "input_path": str(input_path),
        "copied_path": str(copied_path),
        "split": collection.split,
        "num_samples": int(collection.num_samples),
        "layers": collection.layer_names(),
        "metadata": collection.metadata,
    }


def _resolve_class_labels(
    classes: np.ndarray,
    *,
    train_collection: FeatureCollection,
    test_collection: FeatureCollection,
) -> list[str]:
    metadata_candidates = [train_collection.metadata, test_collection.metadata]
    for metadata in metadata_candidates:
        label_names = metadata.get("label_names")
        if isinstance(label_names, list):
            resolved: list[str] = []
            for class_id in classes.tolist():
                if int(class_id) >= len(label_names):
                    resolved = []
                    break
                resolved.append(str(label_names[int(class_id)]))
            if resolved:
                return resolved
        label_map = metadata.get("label_map")
        if isinstance(label_map, dict):
            return [
                str(label_map.get(str(int(class_id)), int(class_id)))
                for class_id in classes.tolist()
            ]
    return [str(int(class_id)) for class_id in classes.tolist()]


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
