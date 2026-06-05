from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _set_nested_value(data: dict[str, Any], path: Sequence[str], value: Any) -> None:
    cursor = data
    for segment in path[:-1]:
        next_value = cursor.get(segment)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[segment] = next_value
        cursor = next_value
    cursor[path[-1]] = deepcopy(value)


def config_digest(data: dict[str, Any], ignore_keys: set[str] | None = None) -> str:
    """Return a stable hash for config-like mappings."""
    ignore_keys = ignore_keys or set()
    filtered = {key: value for key, value in data.items() if key not in ignore_keys}
    payload = json.dumps(_normalize(filtered), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Small config object for experiment identity and metadata."""

    name: str
    dataset: str
    model: str
    seed: int = 0
    tags: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _normalize(asdict(self)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        raw_tags = data.get("tags", ())
        tags: tuple[str, ...]
        if isinstance(raw_tags, str):
            tags = (raw_tags,)
        else:
            tags = tuple(cast(Sequence[str], raw_tags))

        raw_extras = data.get("extras", {})
        extras = cast(dict[str, Any], _normalize(dict(cast(Mapping[str, Any], raw_extras))))

        return cls(
            name=str(data["name"]),
            dataset=str(data["dataset"]),
            model=str(data["model"]),
            seed=int(data.get("seed", 0)),
            tags=tags,
            extras=extras,
        )

    def apply_overrides(self, overrides: Mapping[str, Any]) -> "ExperimentConfig":
        merged = _deep_merge(self.to_dict(), overrides)
        return self.from_dict(merged)

    def with_extras(self, extras: Mapping[str, Any]) -> "ExperimentConfig":
        return self.apply_overrides({"extras": dict(extras)})

    def digest(self, ignore_keys: set[str] | None = None) -> str:
        return config_digest(self.to_dict(), ignore_keys=ignore_keys)

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def expand_experiment_grid(
    base_config: ExperimentConfig,
    sweep: Mapping[str, Sequence[Any]],
) -> list[ExperimentConfig]:
    """
    Expand a cartesian product of overrides into concrete experiment configs.

    Sweep keys may target top-level fields like ``seed`` or nested extras using dotted
    paths such as ``extras.optimizer.lr``.
    """

    if not sweep:
        return [base_config]

    axes = list(sweep.items())
    expanded: list[ExperimentConfig] = []
    for combination in itertools.product(*(values for _, values in axes)):
        payload = base_config.to_dict()
        for (key, _), value in zip(axes, combination, strict=True):
            _set_nested_value(payload, key.split("."), value)
        expanded.append(ExperimentConfig.from_dict(payload))
    return expanded
