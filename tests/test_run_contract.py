from __future__ import annotations

import json
from pathlib import Path

import pytest

from repr_lab.run_contract import (
    RUN_DIRECTORY_CONTRACT_NAME,
    RUN_DIRECTORY_CONTRACT_VERSION,
    RUN_DIRECTORY_KIND,
    RUN_DIRECTORY_PRODUCER_REPO,
    validate_benchmark_result_payload,
    validate_run_manifest_payload,
)


def test_checked_in_run_directory_fixture_matches_contract() -> None:
    fixture_dir = Path(__file__).resolve().parent / "fixtures" / "repr_lab_run_directory_v1"
    manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
    benchmark_result = json.loads(
        (fixture_dir / "benchmark_result.json").read_text(encoding="utf-8")
    )

    validate_run_manifest_payload(manifest)
    validate_benchmark_result_payload(benchmark_result)

    assert manifest["contract_name"] == RUN_DIRECTORY_CONTRACT_NAME
    assert manifest["contract_version"] == RUN_DIRECTORY_CONTRACT_VERSION
    assert manifest["run_directory_kind"] == RUN_DIRECTORY_KIND
    assert manifest["producer_repo"] == RUN_DIRECTORY_PRODUCER_REPO
    assert benchmark_result["run_id"] == manifest["run_id"]


def test_validate_run_manifest_rejects_missing_keys() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        validate_run_manifest_payload({"contract_name": RUN_DIRECTORY_CONTRACT_NAME})


def test_validate_run_manifest_rejects_wrong_contract_name() -> None:
    fixture_dir = Path(__file__).resolve().parent / "fixtures" / "repr_lab_run_directory_v1"
    manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["contract_name"] = "wrong_contract"

    with pytest.raises(ValueError, match="contract_name"):
        validate_run_manifest_payload(manifest)


def test_validate_benchmark_result_rejects_empty_artifact_paths() -> None:
    fixture_dir = Path(__file__).resolve().parent / "fixtures" / "repr_lab_run_directory_v1"
    result = json.loads((fixture_dir / "benchmark_result.json").read_text(encoding="utf-8"))
    result["artifact_paths"] = {}

    with pytest.raises(ValueError, match="must not be empty"):
        validate_benchmark_result_payload(result)
