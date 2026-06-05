from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from repr_lab.features import FeatureCollection


def _as_matrix(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError("features must include a sample axis and at least one feature dimension")
    return array.reshape(array.shape[0], -1)


def _covariance_eigenvalues(features: np.ndarray) -> np.ndarray:
    matrix = _as_matrix(features)
    if matrix.shape[0] <= 1:
        return np.zeros(1, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(matrix.shape[0] - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return np.clip(eigenvalues, a_min=0.0, a_max=None)


class LayerStatistic(ABC):
    """Base interface for per-layer feature statistics."""

    name = "statistic"
    requires_labels = False

    @abstractmethod
    def compute(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> dict[str, float]: ...


class AmbientDimensionStat(LayerStatistic):
    name = "ambient_dimension"

    def compute(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> dict[str, float]:
        matrix = _as_matrix(features)
        return {"ambient_dimension": float(matrix.shape[1])}


class EffectiveRankStat(LayerStatistic):
    name = "effective_rank"

    def compute(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> dict[str, float]:
        eigenvalues = _covariance_eigenvalues(features)
        total = float(eigenvalues.sum())
        if total <= 0:
            return {"effective_rank": 0.0}
        probabilities = eigenvalues / total
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log(probabilities))
        return {"effective_rank": float(np.exp(entropy))}


class ParticipationRatioStat(LayerStatistic):
    name = "participation_ratio"

    def compute(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> dict[str, float]:
        eigenvalues = _covariance_eigenvalues(features)
        denominator = float(np.square(eigenvalues).sum())
        if denominator <= 0:
            return {"participation_ratio": 0.0}
        numerator = float(eigenvalues.sum()) ** 2
        return {"participation_ratio": numerator / denominator}


class ClassGeometryStat(LayerStatistic):
    name = "class_geometry"
    requires_labels = True

    def compute(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> dict[str, float]:
        if labels is None:
            raise ValueError("labels are required for class geometry statistics")
        matrix = _as_matrix(features)
        label_array = np.asarray(labels)
        if label_array.shape[0] != matrix.shape[0]:
            raise ValueError("labels must align with the feature sample count")

        unique_labels = np.unique(label_array)
        if unique_labels.shape[0] <= 1:
            return {
                "centroid_distance": 0.0,
                "within_class_spread": 0.0,
                "separation_ratio": 0.0,
            }

        centroids: list[np.ndarray] = []
        within_spreads: list[float] = []
        for label in unique_labels:
            class_features = matrix[label_array == label]
            centroid = class_features.mean(axis=0)
            centroids.append(centroid)
            deltas = class_features - centroid
            within_spreads.append(float(np.sqrt(np.square(deltas).sum(axis=1).mean())))

        centroid_matrix = np.stack(centroids, axis=0)
        distances: list[float] = []
        for left in range(centroid_matrix.shape[0]):
            for right in range(left + 1, centroid_matrix.shape[0]):
                delta = centroid_matrix[left] - centroid_matrix[right]
                distances.append(float(np.linalg.norm(delta)))

        centroid_distance = float(np.mean(distances))
        within_class_spread = float(np.mean(within_spreads))
        separation_ratio = (
            centroid_distance / within_class_spread if within_class_spread > 0 else 0.0
        )
        return {
            "centroid_distance": centroid_distance,
            "within_class_spread": within_class_spread,
            "separation_ratio": separation_ratio,
        }


@dataclass(frozen=True, slots=True)
class LayerwiseAnalysisResult:
    """Serializable layerwise metric report."""

    split: str
    layers: dict[str, dict[str, float]]
    statistics: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "split": self.split,
            "statistics": list(self.statistics),
            "layers": self.layers,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


class LayerwiseAnalyzer:
    """Apply a set of statistics to each persisted feature layer."""

    def __init__(self, statistics: list[LayerStatistic]) -> None:
        if not statistics:
            raise ValueError("at least one statistic is required")
        self.statistics = statistics

    def analyze(self, collection: FeatureCollection) -> LayerwiseAnalysisResult:
        layer_results: dict[str, dict[str, float]] = {}
        for layer_name in collection.layer_names():
            features = collection.flatten_layer(layer_name)
            layer_metrics: dict[str, float] = {}
            for statistic in self.statistics:
                if statistic.requires_labels and collection.labels is None:
                    raise ValueError(
                        f"Statistic '{statistic.name}' requires labels but the collection has none"
                    )
                layer_metrics.update(statistic.compute(features, collection.labels))
            layer_results[layer_name] = layer_metrics

        return LayerwiseAnalysisResult(
            split=collection.split,
            layers=layer_results,
            statistics=tuple(stat.name for stat in self.statistics),
            metadata=collection.metadata,
        )
