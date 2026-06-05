from __future__ import annotations

import json
from pathlib import Path

import pytest

from repr_lab.main import main
from repr_lab.published_dataset import load_published_release


def test_load_published_release_reads_manifest_and_counts(release_bundle: Path) -> None:
    release = load_published_release(release_bundle)

    assert release.release_id == "demo-release"
    assert release.task_types == ("bbox",)
    assert release.counts["annotation_count"] == 1
    assert release.split_summary == {"unspecified": 1}


def test_inspect_release_cli_prints_summary(release_bundle: Path, capsys) -> None:
    exit_code = main(["inspect-release", "--bundle-dir", str(release_bundle)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["release_id"] == "demo-release"
    assert payload["counts"]["asset_count"] == 1


def test_load_published_release_raises_on_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_published_release(tmp_path / "nonexistent")


def test_load_published_release_rejects_wrong_contract_name(release_bundle: Path) -> None:
    manifest_path = release_bundle / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contract_name"] = "wrong_contract"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported contract_name"):
        load_published_release(release_bundle)


def test_load_published_release_rejects_missing_required_keys(release_bundle: Path) -> None:
    manifest_path = release_bundle / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["release_id"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        load_published_release(release_bundle)
