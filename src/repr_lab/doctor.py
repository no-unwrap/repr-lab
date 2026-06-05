from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from enum import Enum
from typing import Any

from repr_lab.checkpointing import CheckpointSourceKind, inspect_huggingface_cli
from repr_lab.materialization import (
    LOCAL_REFERENCE_MODEL_FAMILIES,
    OPTIONAL_RAW_MEDIA_RUNTIME_PACKAGES,
    optional_raw_media_runtime_contract,
)
from repr_lab.model_catalog import list_model_families


class CheckStatus(str, Enum):
    OK = "ok"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: str
    status: CheckStatus
    detail: str
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
        }
        if self.data:
            payload["data"] = self.data
        return payload


@dataclass(frozen=True, slots=True)
class DoctorReport:
    checks: tuple[DoctorCheck, ...]

    @property
    def overall_status(self) -> CheckStatus:
        if any(check.status is CheckStatus.ERROR for check in self.checks):
            return CheckStatus.ERROR
        if any(check.status is CheckStatus.WARN for check in self.checks):
            return CheckStatus.WARN
        return CheckStatus.OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "checks": [check.to_dict() for check in self.checks],
        }


def run_doctor() -> DoctorReport:
    checks: list[DoctorCheck] = []

    hf_status = inspect_huggingface_cli()
    if not hf_status.available:
        checks.append(
            DoctorCheck(
                name="huggingface-cli",
                status=CheckStatus.WARN,
                detail=hf_status.detail or "Hugging Face CLI is unavailable.",
                data=hf_status.to_dict(),
            )
        )
    elif hf_status.logged_in:
        checks.append(
            DoctorCheck(
                name="huggingface-cli",
                status=CheckStatus.OK,
                detail=f"Logged into Hugging Face as {hf_status.username}.",
                data=hf_status.to_dict(),
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="huggingface-cli",
                status=CheckStatus.WARN,
                detail=hf_status.detail or "Hugging Face CLI is not logged in.",
                data=hf_status.to_dict(),
            )
        )

    hf_families = [
        family.name
        for family in list_model_families()
        if any(
            variant.checkpoint is not None
            and variant.checkpoint.source_kind is CheckpointSourceKind.HUGGINGFACE
            for variant in family.variants
        )
    ]
    if hf_families:
        checks.append(
            DoctorCheck(
                name="huggingface-backed-models",
                status=CheckStatus.OK if hf_status.logged_in else CheckStatus.WARN,
                detail=(
                    "All Hugging Face-backed families are eligible for automated "
                    "checkpoint preparation."
                    if hf_status.logged_in
                    else "Some Hugging Face-backed families are cataloged, but "
                    "automated preparation is blocked until login."
                ),
                data={"families": hf_families},
            )
        )

    local_reference_families = list(LOCAL_REFERENCE_MODEL_FAMILIES)
    checks.append(
        DoctorCheck(
            name="local-reference-paths",
            status=CheckStatus.OK,
            detail=(
                "Some runnable paths remain intentionally local-only: use local "
                "checkpoint manifests and local feature exports rather than "
                "assuming generalized raw-media automation."
            ),
            data={"families": local_reference_families},
        )
    )

    runtime_contract = optional_raw_media_runtime_contract()
    package_status = {
        package: bool(importlib.util.find_spec(package))
        for package in OPTIONAL_RAW_MEDIA_RUNTIME_PACKAGES
    }
    missing_packages = [package for package, available in package_status.items() if not available]
    checks.append(
        DoctorCheck(
            name="optional-runtime-packages",
            status=CheckStatus.OK if not missing_packages else CheckStatus.WARN,
            detail=(
                "Optional raw-media runtime packages are installed."
                if not missing_packages
                else "Some optional raw-media runtime packages are missing; "
                "keep broader adapter work out of scope until they are present."
            ),
            data={
                "contract_name": runtime_contract.contract_name,
                "contract_version": runtime_contract.contract_version,
                "packages": package_status,
                "missing": missing_packages,
            },
        )
    )

    return DoctorReport(checks=tuple(checks))
