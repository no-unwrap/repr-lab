"""Shared JSON loading and coercion helpers used across repr-lab modules."""

from __future__ import annotations

import json
from pathlib import Path


def load_json_object(path: Path) -> dict[str, object]:
    """Read a JSON file and return its top-level object, failing on non-dict payloads."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return {str(key): value for key, value in payload.items()}


def coerce_dict(value: object) -> dict[str, object]:
    """Coerce a parsed JSON value to a string-keyed dict, failing on non-dict input."""
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return {str(key): inner_value for key, inner_value in value.items()}
