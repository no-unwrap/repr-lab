from __future__ import annotations

import json
from pathlib import Path

import pytest

from repr_lab._parsing import coerce_dict, load_json_object


def test_load_json_object_returns_string_keyed_dict(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"key": "value", "count": 3}), encoding="utf-8")

    result = load_json_object(path)

    assert result == {"key": "value", "count": 3}
    assert all(isinstance(k, str) for k in result)


def test_load_json_object_rejects_non_dict_json(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected JSON object"):
        load_json_object(path)


def test_load_json_object_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_json_object(tmp_path / "nonexistent.json")


def test_coerce_dict_returns_string_keyed_dict() -> None:
    result = coerce_dict({"alpha": 1, "beta": "two"})

    assert result == {"alpha": 1, "beta": "two"}
    assert all(isinstance(k, str) for k in result)


def test_coerce_dict_rejects_non_dict_input() -> None:
    with pytest.raises(ValueError, match="Expected JSON object"):
        coerce_dict([1, 2, 3])

    with pytest.raises(ValueError, match="Expected JSON object"):
        coerce_dict("not a dict")

    with pytest.raises(ValueError, match="Expected JSON object"):
        coerce_dict(None)
