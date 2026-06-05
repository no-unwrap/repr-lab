from __future__ import annotations

RUN_DIRECTORY_CONTRACT_NAME = "repr_lab_run_directory_contract"
RUN_DIRECTORY_CONTRACT_VERSION = "1.0.0"
RUN_DIRECTORY_KIND = "benchmark_run_directory"
RUN_DIRECTORY_PRODUCER_REPO = "repr-lab"

REQUIRED_MANIFEST_KEYS = (
    "contract_name",
    "contract_version",
    "run_directory_kind",
    "producer_repo",
    "name",
    "dataset",
    "model",
    "seed",
    "run_id",
    "created_at",
)

REQUIRED_BENCHMARK_RESULT_KEYS = (
    "contract_name",
    "contract_version",
    "run_directory_kind",
    "producer_repo",
    "schema_version",
    "run_id",
    "release_id",
    "release_version",
    "benchmark_task",
    "model_family",
    "model_variant",
    "status",
    "started_at",
    "artifact_paths",
)


def contract_fields() -> dict[str, str]:
    return {
        "contract_name": RUN_DIRECTORY_CONTRACT_NAME,
        "contract_version": RUN_DIRECTORY_CONTRACT_VERSION,
        "run_directory_kind": RUN_DIRECTORY_KIND,
        "producer_repo": RUN_DIRECTORY_PRODUCER_REPO,
    }


def validate_run_manifest_payload(payload: dict[str, object]) -> None:
    _require_keys(payload, REQUIRED_MANIFEST_KEYS, label="run manifest")
    _validate_shared_contract_fields(payload, label="run manifest")


def validate_benchmark_result_payload(payload: dict[str, object]) -> None:
    _require_keys(payload, REQUIRED_BENCHMARK_RESULT_KEYS, label="benchmark_result")
    _validate_shared_contract_fields(payload, label="benchmark_result")
    artifact_paths = payload.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        raise ValueError("benchmark_result artifact_paths must be an object")
    if not artifact_paths:
        raise ValueError("benchmark_result artifact_paths must not be empty")
    for key, value in artifact_paths.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("benchmark_result artifact_paths keys must be non-empty strings")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("benchmark_result artifact_paths values must be non-empty strings")


def _validate_shared_contract_fields(payload: dict[str, object], *, label: str) -> None:
    for field_name, expected_value in contract_fields().items():
        actual_value = payload.get(field_name)
        if actual_value != expected_value:
            raise ValueError(
                f"{label} {field_name} {actual_value!r} does not match expected {expected_value!r}"
            )


def _require_keys(payload: dict[str, object], keys: tuple[str, ...], *, label: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{label} missing required keys: {', '.join(sorted(missing))}")
