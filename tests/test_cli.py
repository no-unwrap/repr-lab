from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from repr_lab import CheckStatus, DoctorCheck, DoctorReport
from repr_lab.main import main


def test_main_without_args_prints_help(capsys) -> None:
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage:" in captured.out.lower()


def test_main_version_flag_prints_version(capsys) -> None:
    exit_code = main(["--version"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "repr-lab 0.1.0" in captured.out


def test_list_models_prints_seeded_catalog(capsys) -> None:
    exit_code = main(["list-models"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "vjepa2" in captured.out
    assert "sam2" in captured.out
    assert "sam3" in captured.out
    assert "segment-anything" in captured.out


def test_prepare_checkpoint_requires_automation_support(capsys) -> None:
    exit_code = 0
    try:
        main(["prepare-checkpoint", "--model", "sam1"])
    except RuntimeError as exc:
        exit_code = 1
        assert "only implemented for Hugging Face sources" in str(exc)

    assert exit_code == 1


def test_execute_release_plan_command_prints_record(monkeypatch, capsys) -> None:
    @dataclass(frozen=True)
    class FakeRecord:
        payload: dict[str, object]

        def to_dict(self) -> dict[str, object]:
            return self.payload

    monkeypatch.setattr(
        "repr_lab.main.execute_release_plan",
        lambda *args, **kwargs: FakeRecord(
            {
                "run_id": "demo-run",
                "status": "succeeded",
            }
        ),
    )

    exit_code = main(
        [
            "execute-release-plan",
            "--plan",
            "artifacts/planned/demo-run",
            "--train-features",
            "/tmp/train_features.npz",
            "--test-features",
            "/tmp/test_features.npz",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "demo-run" in captured.out
    assert "succeeded" in captured.out


def test_record_local_checkpoint_command_prints_manifest(monkeypatch, capsys) -> None:
    @dataclass(frozen=True)
    class FakeManifest:
        payload: dict[str, object]

        def to_dict(self) -> dict[str, object]:
            return self.payload

    monkeypatch.setattr(
        "repr_lab.main.record_local_checkpoint",
        lambda *args, **kwargs: FakeManifest(
            {
                "model_family": "vjepa2",
                "model_variant": "vjepa2_1_vit_base_384",
                "local_only": True,
            }
        ),
    )

    exit_code = main(
        [
            "record-local-checkpoint",
            "--model",
            "vjepa2",
            "--local-path",
            "/tmp/vjepa2",
            "--output",
            "/tmp/checkpoint_manifest.json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "vjepa2" in captured.out
    assert "local_only" in captured.out


def test_materialize_release_features_command_prints_report(monkeypatch, capsys) -> None:
    @dataclass(frozen=True)
    class FakeMaterialized:
        payload: dict[str, object]

        def to_dict(self) -> dict[str, object]:
            return self.payload

    monkeypatch.setattr(
        "repr_lab.main.materialize_release_features",
        lambda *args, **kwargs: FakeMaterialized(
            {
                "run_id": "demo-run",
                "outputs": {
                    "train": str(Path("/tmp/train.npz")),
                    "test": str(Path("/tmp/test.npz")),
                },
            }
        ),
    )

    exit_code = main(
        [
            "materialize-release-features",
            "--plan",
            "artifacts/planned/demo-run",
            "--feature-export",
            "/tmp/export.npz",
            "--checkpoint-manifest",
            "/tmp/checkpoint_manifest.json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "demo-run" in captured.out
    assert "/tmp/train.npz" in captured.out


def test_inspect_vjepa_export_command_prints_report(monkeypatch, capsys) -> None:
    @dataclass(frozen=True)
    class FakeExport:
        payload: dict[str, object]

        def to_dict(self) -> dict[str, object]:
            return self.payload

    monkeypatch.setattr(
        "repr_lab.main.VJEPALocalRawMediaExport.load",
        lambda path: FakeExport(
            {
                "contract_name": "repr_lab_vjepa_local_raw_media_export",
                "contract_version": "0.1.0",
                "compatibility_mode": "explicit_contract",
            }
        ),
    )

    exit_code = main(
        [
            "inspect-vjepa-export",
            "--feature-export",
            "/tmp/export.npz",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "repr_lab_vjepa_local_raw_media_export" in captured.out
    assert "explicit_contract" in captured.out


def test_doctor_command_prints_report(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "repr_lab.main.run_doctor",
        lambda: DoctorReport(
            checks=(
                DoctorCheck(
                    name="huggingface-cli",
                    status=CheckStatus.WARN,
                    detail="Not logged in",
                ),
            )
        ),
    )
    exit_code = main(["doctor"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "overall_status" in captured.out
