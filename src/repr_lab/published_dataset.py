from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from repr_lab._parsing import coerce_dict, load_json_object

PRIMARY_CONTRACT_NAME = "published_artifact_bundle_contract"
SUPPORTED_CONTRACT_NAMES = {PRIMARY_CONTRACT_NAME}
PRIMARY_BUNDLE_KIND = "dataset_release_bundle"
SUPPORTED_BUNDLE_KINDS = {PRIMARY_BUNDLE_KIND}
SUPPORTED_MAJOR_VERSION = "1"
REQUIRED_MANIFEST_KEYS = (
    "contract_name",
    "contract_version",
    "bundle_kind",
    "release_id",
    "release_version",
    "publisher_repo",
    "publisher_commit_sha",
    "canonical_annotation_format",
    "source_formats",
    "task_types",
    "artifact_paths",
    "lineage",
    "split_summary",
)
REQUIRED_ARTIFACT_KEYS = (
    "annotations_coco",
    "assets_table",
    "categories_table",
)


@dataclass(frozen=True, slots=True)
class PublishedRelease:
    bundle_dir: Path
    manifest_path: Path
    release_id: str
    release_version: str
    publisher_repo: str
    publisher_commit_sha: str
    task_types: tuple[str, ...]
    source_formats: tuple[str, ...]
    split_summary: dict[str, int]
    counts: dict[str, int]
    artifact_paths: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_dir": str(self.bundle_dir),
            "manifest_path": str(self.manifest_path),
            "release_id": self.release_id,
            "release_version": self.release_version,
            "publisher_repo": self.publisher_repo,
            "publisher_commit_sha": self.publisher_commit_sha,
            "task_types": list(self.task_types),
            "source_formats": list(self.source_formats),
            "split_summary": self.split_summary,
            "counts": self.counts,
            "artifact_paths": self.artifact_paths,
        }


def load_published_release(bundle_dir: str | Path) -> PublishedRelease:
    resolved_bundle_dir = Path(bundle_dir).resolve()
    manifest_path = resolved_bundle_dir / "release_manifest.json"
    manifest = load_json_object(manifest_path)
    _require_keys(manifest, REQUIRED_MANIFEST_KEYS)
    _validate_contract(manifest)

    artifact_paths_raw = coerce_dict(manifest["artifact_paths"])
    _require_keys(artifact_paths_raw, REQUIRED_ARTIFACT_KEYS)
    resolved_artifact_paths = {
        key: str((resolved_bundle_dir / str(value)).resolve())
        for key, value in artifact_paths_raw.items()
    }
    for key in REQUIRED_ARTIFACT_KEYS:
        artifact_path = Path(resolved_artifact_paths[key])
        if not artifact_path.exists():
            raise ValueError(f"missing declared artifact for {key}: {artifact_path}")

    counts = _derive_counts(Path(resolved_artifact_paths["annotations_coco"]))
    split_summary = {
        str(key): _coerce_int(value)
        for key, value in coerce_dict(manifest["split_summary"]).items()
    }
    return PublishedRelease(
        bundle_dir=resolved_bundle_dir,
        manifest_path=manifest_path,
        release_id=str(manifest["release_id"]),
        release_version=str(manifest["release_version"]),
        publisher_repo=str(manifest["publisher_repo"]),
        publisher_commit_sha=str(manifest["publisher_commit_sha"]),
        task_types=tuple(str(item) for item in _coerce_list(manifest["task_types"])),
        source_formats=tuple(str(item) for item in _coerce_list(manifest["source_formats"])),
        split_summary=split_summary,
        counts=counts,
        artifact_paths=resolved_artifact_paths,
    )


def _validate_contract(manifest: dict[str, object]) -> None:
    if str(manifest["contract_name"]) not in SUPPORTED_CONTRACT_NAMES:
        raise ValueError("unsupported contract_name")
    if str(manifest["bundle_kind"]) not in SUPPORTED_BUNDLE_KINDS:
        raise ValueError("unsupported bundle_kind")
    contract_version = str(manifest["contract_version"])
    if contract_version.split(".", 1)[0] != SUPPORTED_MAJOR_VERSION:
        raise ValueError("unsupported contract_version major")


def _derive_counts(annotations_path: Path) -> dict[str, int]:
    payload = load_json_object(annotations_path)
    images = _coerce_list(payload.get("images"))
    annotations = _coerce_list(payload.get("annotations"))
    categories = _coerce_list(payload.get("categories"))
    return {
        "asset_count": len(images),
        "annotation_count": len(annotations),
        "category_count": len(categories),
    }


def _require_keys(payload: dict[str, object], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"missing required keys: {', '.join(sorted(missing))}")


def _coerce_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected list value")
    return list(value)


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    raise ValueError("expected integer-compatible value")
