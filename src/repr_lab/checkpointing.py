from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from repr_lab._parsing import coerce_dict, load_json_object


class ExecutionMaturity(str, Enum):
    CATALOG_ONLY = "catalog_only"
    PLANNED = "planned"
    RUNNABLE = "runnable"
    VALIDATED = "validated"


class CheckpointSourceKind(str, Enum):
    HUGGINGFACE = "huggingface"
    TORCH_HUB = "torch_hub"
    REPO_RELEASE = "repo_release"
    DIRECT_URL = "direct_url"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class HuggingFaceCLIStatus:
    available: bool
    executable: str | None
    logged_in: bool
    username: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "available": self.available,
            "logged_in": self.logged_in,
        }
        if self.executable is not None:
            payload["executable"] = self.executable
        if self.username is not None:
            payload["username"] = self.username
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    source_kind: CheckpointSourceKind
    locator: str
    repo_type: str = "model"
    revision: str | None = None
    filenames: tuple[str, ...] = ()
    include_patterns: tuple[str, ...] = ()
    gated: bool = False
    license: str | None = None
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CheckpointSpec":
        return cls(
            source_kind=CheckpointSourceKind(str(payload["source_kind"])),
            locator=str(payload["locator"]),
            repo_type=str(payload.get("repo_type", "model")),
            revision=str(payload["revision"]) if payload.get("revision") is not None else None,
            filenames=_coerce_string_tuple(payload.get("filenames")),
            include_patterns=_coerce_string_tuple(payload.get("include_patterns")),
            gated=bool(payload.get("gated", False)),
            license=str(payload["license"]) if payload.get("license") is not None else None,
            notes=_coerce_string_tuple(payload.get("notes")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_kind": self.source_kind.value,
            "locator": self.locator,
            "repo_type": self.repo_type,
            "gated": self.gated,
        }
        if self.revision is not None:
            payload["revision"] = self.revision
        if self.filenames:
            payload["filenames"] = list(self.filenames)
        if self.include_patterns:
            payload["include_patterns"] = list(self.include_patterns)
        if self.license is not None:
            payload["license"] = self.license
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True, slots=True)
class PreparedCheckpoint:
    spec: CheckpointSpec
    local_dir: Path
    prepared_at: str
    resolved_revision: str | None
    command: tuple[str, ...]
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "spec": self.spec.to_dict(),
            "local_dir": str(self.local_dir),
            "prepared_at": self.prepared_at,
            "command": list(self.command),
            "files": list(self.files),
        }
        if self.resolved_revision is not None:
            payload["resolved_revision"] = self.resolved_revision
        return payload

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class ResolvedCheckpoint:
    spec: CheckpointSpec
    local_path: Path
    resolved_at: str
    resolved_revision: str | None = None
    files: tuple[str, ...] = ()
    command: tuple[str, ...] = ()
    local_only: bool = True
    model_family: str | None = None
    model_variant: str | None = None
    notes: tuple[str, ...] = ()

    @classmethod
    def from_prepared(
        cls,
        prepared: PreparedCheckpoint,
        *,
        model_family: str | None = None,
        model_variant: str | None = None,
        local_only: bool = False,
        notes: tuple[str, ...] = (),
    ) -> "ResolvedCheckpoint":
        return cls(
            spec=prepared.spec,
            local_path=prepared.local_dir,
            resolved_at=prepared.prepared_at,
            resolved_revision=prepared.resolved_revision,
            files=prepared.files,
            command=prepared.command,
            local_only=local_only,
            model_family=model_family,
            model_variant=model_variant,
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "0.1.0",
            "spec": self.spec.to_dict(),
            "local_path": str(self.local_path),
            "resolved_at": self.resolved_at,
            "local_only": self.local_only,
        }
        if self.resolved_revision is not None:
            payload["resolved_revision"] = self.resolved_revision
        if self.files:
            payload["files"] = list(self.files)
        if self.command:
            payload["command"] = list(self.command)
        if self.model_family is not None:
            payload["model_family"] = self.model_family
        if self.model_variant is not None:
            payload["model_variant"] = self.model_variant
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload

    def to_feature_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_kind": self.spec.source_kind.value,
            "locator": self.spec.locator,
            "repo_type": self.spec.repo_type,
            "local_path": str(self.local_path),
            "local_only": self.local_only,
        }
        if self.resolved_revision is not None:
            payload["resolved_revision"] = self.resolved_revision
        if self.files:
            payload["files"] = list(self.files)
        if self.model_family is not None:
            payload["model_family"] = self.model_family
        if self.model_variant is not None:
            payload["model_variant"] = self.model_variant
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def inspect_huggingface_cli() -> HuggingFaceCLIStatus:
    cli = shutil.which("huggingface-cli") or shutil.which("hf")
    if cli is None:
        return HuggingFaceCLIStatus(
            available=False,
            executable=None,
            logged_in=False,
            detail="Neither 'huggingface-cli' nor 'hf' is available on PATH.",
        )

    result = subprocess.run([cli, "whoami"], capture_output=True, text=True)
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode == 0 and stdout and stdout != "Not logged in":
        return HuggingFaceCLIStatus(
            available=True,
            executable=cli,
            logged_in=True,
            username=stdout,
        )

    detail = stdout or stderr or "Unable to determine Hugging Face CLI login state."
    return HuggingFaceCLIStatus(
        available=True,
        executable=cli,
        logged_in=False,
        detail=detail,
    )


def prepare_checkpoint(
    spec: CheckpointSpec,
    output_dir: str | Path,
    *,
    revision: str | None = None,
    filenames: tuple[str, ...] = (),
    include_patterns: tuple[str, ...] = (),
    force: bool = False,
) -> PreparedCheckpoint:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if spec.source_kind is not CheckpointSourceKind.HUGGINGFACE:
        raise RuntimeError(
            "Automated checkpoint preparation is only implemented for "
            f"Hugging Face sources, got '{spec.source_kind.value}'."
        )

    cli_status = inspect_huggingface_cli()
    if not cli_status.available or cli_status.executable is None:
        raise RuntimeError(cli_status.detail or "Hugging Face CLI is unavailable.")
    if spec.gated and not cli_status.logged_in:
        raise RuntimeError(
            "Gated Hugging Face checkpoint preparation requires a logged-in "
            f"CLI session. Current state: {cli_status.detail or 'not logged in'}"
        )
    cli = cli_status.executable

    command = [
        cli,
        "download",
        spec.locator,
        "--repo-type",
        spec.repo_type,
        "--local-dir",
        str(output_path),
    ]
    resolved_revision = revision or spec.revision
    if resolved_revision is not None:
        command.extend(["--revision", resolved_revision])
    requested_files = tuple(filenames) or spec.filenames
    if requested_files:
        command.extend(requested_files)
    requested_patterns = tuple(include_patterns) or spec.include_patterns
    if requested_patterns:
        command.extend(["--include", *requested_patterns])
    if force:
        command.append("--force-download")

    subprocess.run(command, check=True)
    prepared_files = tuple(
        sorted(
            str(path.relative_to(output_path))
            for path in output_path.rglob("*")
            if path.is_file() and path.name != "checkpoint.json"
        )
    )
    return PreparedCheckpoint(
        spec=spec,
        local_dir=output_path,
        prepared_at=datetime.now(timezone.utc).isoformat(),
        resolved_revision=resolved_revision,
        command=tuple(command),
        files=prepared_files,
    )


def prepare_model_checkpoint(
    model_family: str,
    *,
    variant: str | None = None,
    output_root: str | Path = "artifacts/checkpoints",
    revision: str | None = None,
    filenames: tuple[str, ...] = (),
    include_patterns: tuple[str, ...] = (),
    force: bool = False,
) -> PreparedCheckpoint:
    from repr_lab.model_catalog import resolve_model_family

    family = resolve_model_family(model_family)
    resolved_variant = family.resolve_variant(variant)
    if resolved_variant.checkpoint is None:
        raise RuntimeError(
            f"Model variant '{resolved_variant.name}' does not define checkpoint provenance."
        )
    output_dir = Path(output_root) / family.name / resolved_variant.name
    prepared = prepare_checkpoint(
        resolved_variant.checkpoint,
        output_dir,
        revision=revision,
        filenames=filenames,
        include_patterns=include_patterns,
        force=force,
    )
    prepared.write_json(output_dir / "checkpoint.json")
    return prepared


def load_checkpoint_manifest(path: str | Path) -> ResolvedCheckpoint:
    payload = load_json_object(Path(path))
    if "prepared_at" in payload and "local_dir" in payload:
        prepared = PreparedCheckpoint(
            spec=CheckpointSpec.from_dict(coerce_dict(payload.get("spec"))),
            local_dir=Path(str(payload["local_dir"])).resolve(),
            prepared_at=str(payload["prepared_at"]),
            resolved_revision=(
                str(payload["resolved_revision"])
                if payload.get("resolved_revision") is not None
                else None
            ),
            command=_coerce_string_tuple(payload.get("command")),
            files=_coerce_string_tuple(payload.get("files")),
        )
        return ResolvedCheckpoint.from_prepared(
            prepared,
            model_family=str(payload["model_family"]) if payload.get("model_family") else None,
            model_variant=str(payload["model_variant"]) if payload.get("model_variant") else None,
            local_only=bool(payload.get("local_only", False)),
            notes=_coerce_string_tuple(payload.get("notes")),
        )

    return ResolvedCheckpoint(
        spec=CheckpointSpec.from_dict(coerce_dict(payload.get("spec"))),
        local_path=Path(str(payload["local_path"])).resolve(),
        resolved_at=str(payload["resolved_at"]),
        resolved_revision=(
            str(payload["resolved_revision"])
            if payload.get("resolved_revision") is not None
            else None
        ),
        files=_coerce_string_tuple(payload.get("files")),
        command=_coerce_string_tuple(payload.get("command")),
        local_only=bool(payload.get("local_only", True)),
        model_family=str(payload["model_family"]) if payload.get("model_family") else None,
        model_variant=str(payload["model_variant"]) if payload.get("model_variant") else None,
        notes=_coerce_string_tuple(payload.get("notes")),
    )


def record_local_checkpoint(
    model_family: str,
    *,
    variant: str | None = None,
    local_path: str | Path,
    output_path: str | Path,
    revision: str | None = None,
    files: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> ResolvedCheckpoint:
    from repr_lab.model_catalog import resolve_model_family

    family = resolve_model_family(model_family)
    resolved_variant = family.resolve_variant(variant)
    if resolved_variant.checkpoint is None:
        raise RuntimeError(
            f"Model variant '{resolved_variant.name}' does not define checkpoint provenance."
        )

    base_spec = resolved_variant.checkpoint
    manifest = ResolvedCheckpoint(
        spec=CheckpointSpec(
            source_kind=base_spec.source_kind,
            locator=base_spec.locator,
            repo_type=base_spec.repo_type,
            revision=revision or base_spec.revision,
            filenames=base_spec.filenames,
            include_patterns=base_spec.include_patterns,
            gated=base_spec.gated,
            license=base_spec.license,
            notes=base_spec.notes,
        ),
        local_path=Path(local_path).resolve(),
        resolved_at=datetime.now(timezone.utc).isoformat(),
        resolved_revision=revision or base_spec.revision,
        files=tuple(files),
        local_only=True,
        model_family=family.name,
        model_variant=resolved_variant.name,
        notes=tuple(notes),
    )
    manifest.write_json(output_path)
    return manifest


def _coerce_string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("Expected list value")
    return tuple(str(item) for item in value)
