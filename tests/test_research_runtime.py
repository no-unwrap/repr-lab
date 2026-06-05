from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from repr_lab import (
    AmbientDimensionStat,
    ClassGeometryStat,
    DatasetSpec,
    EffectiveRankStat,
    ExperimentConfig,
    ExperimentPaths,
    FeatureCollection,
    LayerwiseAnalyzer,
    ParticipationRatioStat,
    TrainerBase,
    TrainerConfig,
    expand_experiment_grid,
)


class DummyTrainer(TrainerBase):
    def __init__(
        self,
        trainer_config: TrainerConfig,
        *,
        experiment_config: ExperimentConfig,
        paths: ExperimentPaths,
    ) -> None:
        super().__init__(
            trainer_config,
            experiment_config=experiment_config,
            paths=paths,
        )
        self.saved_checkpoints: list[tuple[int, int]] = []
        self.best_events: list[tuple[int, int, float]] = []

    def train_epoch(self, epoch: int) -> dict[str, float]:
        return {"loss": 1.0 / epoch}

    def evaluate_epoch(self, epoch: int) -> dict[str, float]:
        return {"accuracy": float(epoch), "loss": 1.0 / (epoch + 1)}

    def save_checkpoint(self, run_index: int, epoch: int) -> None:
        self.saved_checkpoints.append((run_index, epoch))

    def on_best_checkpoint(self, run_index: int, epoch: int, metrics: dict[str, float]) -> None:
        self.best_events.append((run_index, epoch, metrics["accuracy"]))


def test_experiment_config_supports_nested_overrides_and_grid_expansion() -> None:
    base = ExperimentConfig(
        name="baseline",
        dataset="cifar10",
        model="resnet18",
        seed=0,
        extras={"optimizer": {"lr": 0.1, "weight_decay": 0.0}},
    )

    updated = base.apply_overrides({"extras": {"optimizer": {"weight_decay": 0.01}}})
    assert updated.extras["optimizer"]["lr"] == 0.1
    assert updated.extras["optimizer"]["weight_decay"] == 0.01

    grid = expand_experiment_grid(
        base,
        {
            "seed": [1, 2],
            "extras.optimizer.lr": [0.1, 0.01],
        },
    )

    assert len(grid) == 4
    assert sorted({config.seed for config in grid}) == [1, 2]
    assert sorted({config.extras["optimizer"]["lr"] for config in grid}) == [0.01, 0.1]
    assert len({config.digest() for config in grid}) == 4


def test_dataset_spec_exposes_basic_dataset_contract() -> None:
    spec = DatasetSpec(
        name="cifar10",
        train_size=50_000,
        test_size=10_000,
        num_classes=10,
        input_shape=(3, 32, 32),
    )

    assert spec.total_size == 60_000
    assert spec.batches_per_epoch(128) == 391
    assert spec.batches_per_epoch(250, split="test") == 40


def test_feature_collection_round_trips_and_supports_layerwise_analysis(tmp_path: Path) -> None:
    collection = FeatureCollection(
        layers={
            "stem": np.array(
                [
                    [0.0, 0.0],
                    [0.1, 0.0],
                    [1.0, 1.0],
                    [1.1, 1.0],
                ]
            ),
            "head": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [1.2, 1.0, 1.0],
                ]
            ),
        },
        labels=np.array([0, 0, 1, 1]),
        split="test",
        metadata={"dataset": "toy"},
    )
    output_path = tmp_path / "features_test.npz"
    collection.save(output_path)

    reloaded = FeatureCollection.load(output_path)
    analyzer = LayerwiseAnalyzer(
        [
            AmbientDimensionStat(),
            EffectiveRankStat(),
            ParticipationRatioStat(),
            ClassGeometryStat(),
        ]
    )
    report = analyzer.analyze(reloaded)
    report_path = tmp_path / "analysis.json"
    report.write_json(report_path)

    assert reloaded.layer_names() == ["head", "stem"]
    assert report.layers["stem"]["ambient_dimension"] == 2.0
    assert report.layers["stem"]["effective_rank"] > 0.0
    assert report.layers["stem"]["participation_ratio"] > 0.0
    assert report.layers["stem"]["separation_ratio"] > 1.0
    assert json.loads(report_path.read_text(encoding="utf-8"))["split"] == "test"


def test_trainer_base_runs_multiple_trials_and_persists_result(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="toy-train",
        dataset="toyset",
        model="linear-probe",
        extras={"trainer": {"num_epochs": 3, "num_runs": 2, "checkpoint_interval": 2}},
    )
    paths = ExperimentPaths.from_config(tmp_path, config)
    trainer = DummyTrainer(
        TrainerConfig.from_experiment_config(config),
        experiment_config=config,
        paths=paths,
    )

    callback_epochs: list[tuple[int, int]] = []
    trainer.set_epoch_callback(
        2,
        lambda _record, run_index, epoch: callback_epochs.append((run_index, epoch)),
    )
    summary = trainer.run()

    assert summary.aggregate["mean_best_metric"] == 3.0
    assert [run.best_epoch for run in summary.runs] == [3, 3]
    assert trainer.saved_checkpoints == [(1, 2), (2, 2)]
    assert callback_epochs == [(1, 2), (2, 2)]
    assert len(trainer.best_events) == 6
    assert paths.result_path.exists()
