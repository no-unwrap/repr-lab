from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Metadata contract for a dataset exposed through the framework."""

    name: str
    train_size: int
    test_size: int
    num_classes: int
    input_shape: tuple[int, ...]
    normalization: str = "mean_std"
    label_names: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.train_size <= 0:
            raise ValueError("train_size must be positive")
        if self.test_size <= 0:
            raise ValueError("test_size must be positive")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not self.input_shape:
            raise ValueError("input_shape must be non-empty")
        if self.label_names and len(self.label_names) != self.num_classes:
            raise ValueError("label_names must match num_classes when provided")

    @property
    def total_size(self) -> int:
        return self.train_size + self.test_size

    def batches_per_epoch(self, batch_size: int, *, split: str = "train") -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        size = self.train_size if split == "train" else self.test_size
        return (size + batch_size - 1) // batch_size

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "num_classes": self.num_classes,
            "input_shape": list(self.input_shape),
            "normalization": self.normalization,
        }
        if self.label_names:
            payload["label_names"] = list(self.label_names)
        if self.extras:
            payload["extras"] = self.extras
        return payload
