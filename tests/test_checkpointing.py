from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from repr_lab import (
    CheckpointSourceKind,
    CheckpointSpec,
    PreparedCheckpoint,
    load_checkpoint_manifest,
    prepare_model_checkpoint,
    record_local_checkpoint,
)
from repr_lab.checkpointing import prepare_checkpoint


def test_prepare_checkpoint_uses_huggingface_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        if name == "huggingface-cli":
            return "/usr/local/bin/huggingface-cli"
        return None

    def fake_run(command: list[str], *args, **kwargs):
        commands.append(command)
        if command[1] == "whoami":

            class Result:
                returncode = 0
                stdout = "unit-test-user\n"
                stderr = ""

            return Result()
        (tmp_path / "weights.safetensors").write_text("placeholder", encoding="utf-8")
        return None

    monkeypatch.setattr("repr_lab.checkpointing.shutil.which", fake_which)
    monkeypatch.setattr("repr_lab.checkpointing.subprocess.run", fake_run)

    prepared = prepare_checkpoint(
        CheckpointSpec(
            source_kind=CheckpointSourceKind.HUGGINGFACE,
            locator="facebook/sam3",
            gated=True,
        ),
        tmp_path,
        revision="main",
        filenames=("weights.safetensors",),
        force=True,
    )

    assert commands == [
        ["/usr/local/bin/huggingface-cli", "whoami"],
        [
            "/usr/local/bin/huggingface-cli",
            "download",
            "facebook/sam3",
            "--repo-type",
            "model",
            "--local-dir",
            str(tmp_path),
            "--revision",
            "main",
            "weights.safetensors",
            "--force-download",
        ],
    ]
    assert prepared.files == ("weights.safetensors",)


def test_prepare_model_checkpoint_writes_provenance_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_which(name: str) -> str | None:
        if name == "huggingface-cli":
            return "/usr/local/bin/huggingface-cli"
        return None

    def fake_run(command: list[str], *args, **kwargs):
        if command[1] == "whoami":

            class Result:
                returncode = 0
                stdout = "unit-test-user\n"
                stderr = ""

            return Result()
        target_dir = Path(command[command.index("--local-dir") + 1])
        (target_dir / "weights.safetensors").write_text("placeholder", encoding="utf-8")
        return None

    monkeypatch.setattr("repr_lab.checkpointing.shutil.which", fake_which)
    monkeypatch.setattr("repr_lab.checkpointing.subprocess.run", fake_run)

    prepared = prepare_model_checkpoint("sam3", output_root=tmp_path)
    provenance_path = tmp_path / "sam3" / "sam3" / "checkpoint.json"

    assert prepared.spec.locator == "facebook/sam3"
    assert provenance_path.exists()
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert payload["spec"]["source_kind"] == "huggingface"
    assert payload["spec"]["locator"] == "facebook/sam3"


def test_prepare_checkpoint_requires_login_for_gated_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_which(name: str) -> str | None:
        if name == "huggingface-cli":
            return "/usr/local/bin/huggingface-cli"
        return None

    def fake_run(command: list[str], *args, **kwargs):
        if command[1] == "whoami":

            class Result:
                returncode = 0
                stdout = "Not logged in\n"
                stderr = ""

            return Result()
        raise AssertionError("download should not run when login is missing")

    monkeypatch.setattr("repr_lab.checkpointing.shutil.which", fake_which)
    monkeypatch.setattr("repr_lab.checkpointing.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="logged-in CLI session"):
        prepare_checkpoint(
            CheckpointSpec(
                source_kind=CheckpointSourceKind.HUGGINGFACE,
                locator="facebook/sam3",
                gated=True,
            ),
            tmp_path,
        )


def test_record_local_checkpoint_writes_manifest(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "vjepa2_snapshot"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.pt").write_text("placeholder", encoding="utf-8")
    output_path = tmp_path / "checkpoint_manifest.json"

    manifest = record_local_checkpoint(
        "vjepa2",
        local_path=checkpoint_dir,
        output_path=output_path,
        files=("model.pt",),
        notes=("Local snapshot for bounded probe execution.",),
    )

    assert output_path.exists()
    loaded = load_checkpoint_manifest(output_path)
    assert loaded.local_only is True
    assert loaded.model_family == "vjepa2"
    assert loaded.model_variant == "vjepa2_1_vit_base_384"
    assert loaded.files == ("model.pt",)
    assert loaded.spec.locator == "facebookresearch/vjepa2"
    assert manifest.to_feature_metadata()["local_path"] == str(checkpoint_dir.resolve())


def test_load_checkpoint_manifest_accepts_prepared_checkpoint_shape(tmp_path: Path) -> None:
    prepared = PreparedCheckpoint(
        spec=CheckpointSpec(
            source_kind=CheckpointSourceKind.HUGGINGFACE,
            locator="facebook/sam3",
        ),
        local_dir=tmp_path / "sam3",
        prepared_at=datetime.now(timezone.utc).isoformat(),
        resolved_revision="main",
        command=("huggingface-cli", "download"),
        files=("weights.safetensors",),
    )
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(prepared.to_dict(), indent=2) + "\n", encoding="utf-8")

    loaded = load_checkpoint_manifest(path)

    assert loaded.local_only is False
    assert loaded.local_path == (tmp_path / "sam3").resolve()
    assert loaded.command == ("huggingface-cli", "download")
    assert loaded.files == ("weights.safetensors",)
