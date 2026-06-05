from pathlib import Path

from repr_lab import ExperimentConfig, ExperimentPaths, Registry, config_digest


def test_config_digest_is_stable_across_mapping_order() -> None:
    left = {"model": "vit-b", "dataset": "cifar10", "seed": 7}
    right = {"seed": 7, "dataset": "cifar10", "model": "vit-b"}
    assert config_digest(left) == config_digest(right)


def test_config_digest_can_ignore_keys() -> None:
    data = {"model": "vit-b", "dataset": "cifar10", "notes": "scratch"}
    without_notes = {"model": "vit-b", "dataset": "cifar10"}
    assert config_digest(data, ignore_keys={"notes"}) == config_digest(without_notes)


def test_experiment_paths_initialize_writes_config(tmp_path: Path) -> None:
    config = ExperimentConfig(name="baseline", dataset="cifar10", model="resnet18", seed=3)
    paths = ExperimentPaths.from_config(tmp_path, config).initialize(config)

    assert paths.run_dir.exists()
    assert paths.artifacts_dir.exists()
    assert paths.checkpoints_dir.exists()
    assert paths.analysis_dir.exists()
    assert paths.config_path.exists()
    assert paths.manifest_path.exists()
    assert '"model": "resnet18"' in paths.config_path.read_text(encoding="utf-8")
    assert paths.run_id in paths.manifest_path.read_text(encoding="utf-8")


def test_registry_registers_and_resolves_values() -> None:
    registry = Registry[str]("model")
    registry.register("baseline", "resnet18")

    assert registry.get("baseline") == "resnet18"
    assert registry.names() == ["baseline"]
