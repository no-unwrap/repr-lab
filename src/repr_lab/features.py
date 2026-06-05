from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np


def _to_array_map(layers: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for name, values in layers.items():
        if not name:
            raise ValueError("layer names must be non-empty")
        array = np.asarray(values)
        if array.ndim == 0:
            raise ValueError("layer features must include a sample axis")
        normalized[name] = array
    if not normalized:
        raise ValueError("at least one feature layer is required")
    return normalized


@dataclass(frozen=True, slots=True)
class FeatureCollection:
    """Persisted layer activations for a split of an experiment run."""

    layers: dict[str, np.ndarray]
    labels: np.ndarray | None = None
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_layers = _to_array_map(self.layers)
        sample_counts = {array.shape[0] for array in normalized_layers.values()}
        if len(sample_counts) != 1:
            raise ValueError("all layers must share the same number of samples")

        normalized_labels: np.ndarray | None = None
        if self.labels is not None:
            normalized_labels = np.asarray(self.labels)
            if normalized_labels.ndim != 1:
                raise ValueError("labels must be a 1D array")
            if normalized_labels.shape[0] != next(iter(sample_counts)):
                raise ValueError("labels must align with the feature sample count")

        object.__setattr__(self, "layers", normalized_layers)
        object.__setattr__(self, "labels", normalized_labels)

    @property
    def num_samples(self) -> int:
        return int(next(iter(self.layers.values())).shape[0])

    def layer_names(self) -> list[str]:
        return sorted(self.layers)

    def flatten_layer(self, name: str) -> np.ndarray:
        try:
            layer = self.layers[name]
        except KeyError as exc:
            known = ", ".join(self.layer_names())
            raise KeyError(f"Unknown layer '{name}'. Known layers: {known}") from exc
        return layer.reshape(layer.shape[0], -1).astype(np.float64, copy=False)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, np.ndarray] = {
            "__split__": np.array(self.split),
            "__metadata__": np.array(json.dumps(self.metadata, sort_keys=True)),
            "__layer_names__": np.array(self.layer_names()),
        }
        if self.labels is not None:
            payload["__labels__"] = self.labels
        for name, values in self.layers.items():
            payload[f"layer::{name}"] = values
        np.savez_compressed(output_path, **cast(Any, payload))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureCollection":
        input_path = Path(path)
        with np.load(input_path, allow_pickle=False) as payload:
            layer_names = [str(name) for name in payload["__layer_names__"].tolist()]
            layers = {name: payload[f"layer::{name}"] for name in layer_names}
            labels = payload["__labels__"] if "__labels__" in payload.files else None
            split = str(payload["__split__"].item())
            metadata = json.loads(str(payload["__metadata__"].item()))
        return cls(layers=layers, labels=labels, split=split, metadata=metadata)
