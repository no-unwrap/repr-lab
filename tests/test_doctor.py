from __future__ import annotations

from repr_lab import CheckStatus, run_doctor


def test_doctor_reports_huggingface_login_warning(monkeypatch) -> None:
    class FakeStatus:
        available = True
        executable = "/usr/local/bin/huggingface-cli"
        logged_in = False
        username = None
        detail = "Not logged in"

        def to_dict(self):
            return {
                "available": self.available,
                "executable": self.executable,
                "logged_in": self.logged_in,
                "detail": self.detail,
            }

    monkeypatch.setattr("repr_lab.doctor.inspect_huggingface_cli", lambda: FakeStatus())
    monkeypatch.setattr("repr_lab.doctor.importlib.util.find_spec", lambda name: object())

    report = run_doctor()

    assert report.overall_status is CheckStatus.WARN
    assert report.checks[0].name == "huggingface-cli"
    assert report.checks[0].status is CheckStatus.WARN
    assert report.checks[1].name == "huggingface-backed-models"
    assert report.checks[2].name == "local-reference-paths"
    assert report.checks[3].name == "optional-runtime-packages"
    assert report.checks[3].status is CheckStatus.OK


def test_doctor_reports_optional_runtime_packages_warning(monkeypatch) -> None:
    class FakeStatus:
        available = True
        executable = "/usr/local/bin/huggingface-cli"
        logged_in = True
        username = "unit-test-user"
        detail = None

        def to_dict(self):
            return {
                "available": self.available,
                "executable": self.executable,
                "logged_in": self.logged_in,
                "username": self.username,
            }

    def fake_find_spec(name: str) -> object | None:
        return None if name == "torchvision" else object()

    monkeypatch.setattr("repr_lab.doctor.inspect_huggingface_cli", lambda: FakeStatus())
    monkeypatch.setattr("repr_lab.doctor.importlib.util.find_spec", fake_find_spec)

    report = run_doctor()

    assert report.overall_status is CheckStatus.WARN
    assert report.checks[3].name == "optional-runtime-packages"
    assert report.checks[3].status is CheckStatus.WARN
    assert report.checks[3].data == {
        "contract_name": "repr_lab_optional_raw_media_runtime",
        "contract_version": "0.1.0",
        "packages": {
            "torch": True,
            "torchvision": False,
            "PIL": True,
            "transformers": True,
        },
        "missing": ["torchvision"],
    }
