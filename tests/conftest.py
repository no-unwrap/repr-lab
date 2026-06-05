"""Shared test fixtures for repr-lab."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def write_simple_release_bundle(bundle_dir: Path) -> None:
    """Write a minimal valid published-release bundle for unit tests.

    This is the single-image, single-category variant used by tests that
    need *any* valid bundle but do not depend on specific annotation
    geometry or split structure.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "annotations.coco.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "scene.jpg"}],
                "categories": [{"id": 7, "name": "signal-object"}],
                "annotations": [{"id": 10, "image_id": 1, "category_id": 7, "bbox": [1, 2, 3, 4]}],
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "assets.parquet").write_text("parquet-placeholder", encoding="utf-8")
    (bundle_dir / "categories.parquet").write_text("parquet-placeholder", encoding="utf-8")
    (bundle_dir / "release_manifest.json").write_text(
        json.dumps(
            {
                "contract_name": "published_artifact_bundle_contract",
                "contract_version": "1.0.0",
                "bundle_kind": "dataset_release_bundle",
                "release_id": "demo-release",
                "release_version": "1.0.0",
                "publisher_repo": "label-lab",
                "publisher_commit_sha": "abc123",
                "canonical_annotation_format": "COCO",
                "source_formats": ["COCO"],
                "task_types": ["bbox"],
                "artifact_paths": {
                    "annotations_coco": "annotations.coco.json",
                    "assets_table": "assets.parquet",
                    "categories_table": "categories.parquet",
                },
                "lineage": [],
                "split_summary": {"unspecified": 1},
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture()
def release_bundle(tmp_path: Path) -> Path:
    """Return the path to a minimal valid published-release bundle."""
    bundle_dir = tmp_path / "bundle"
    write_simple_release_bundle(bundle_dir)
    return bundle_dir
