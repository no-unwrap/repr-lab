from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from time import perf_counter
from typing import Any

from repr_lab.config import ExperimentConfig
from repr_lab.experiment import ExperimentPaths

MetricMap = Mapping[str, float | int]
EpochCallback = Callable[["EpochRecord", int, int], None]
StopCondition = Callable[[list["EpochRecord"], int, int], bool]


def _coerce_metric_map(metrics: MetricMap) -> dict[str, float]:
    if not metrics:
        raise ValueError("metric mappings must be non-empty")
    return {str(name): float(value) for name, value in metrics.items()}


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Framework-level training loop controls inspired by Arch's ITrainer."""

    num_epochs: int
    num_runs: int = 1
    eval_interval: int = 1
    checkpoint_interval: int | None = None
    selection_metric: str = "accuracy"
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.num_runs <= 0:
            raise ValueError("num_runs must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.checkpoint_interval is not None and self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive when provided")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrainerConfig":
        return cls(
            num_epochs=int(data["num_epochs"]),
            num_runs=int(data.get("num_runs", 1)),
            eval_interval=int(data.get("eval_interval", 1)),
            checkpoint_interval=(
                int(data["checkpoint_interval"]) if data.get("checkpoint_interval") else None
            ),
            selection_metric=str(data.get("selection_metric", "accuracy")),
            higher_is_better=bool(data.get("higher_is_better", True)),
        )

    @classmethod
    def from_experiment_config(
        cls,
        config: ExperimentConfig,
        *,
        section: str = "trainer",
    ) -> "TrainerConfig":
        raw_section = config.extras.get(section, {})
        if not isinstance(raw_section, Mapping):
            raise ValueError(f"ExperimentConfig.extras['{section}'] must be a mapping")
        return cls.from_mapping(raw_section)


@dataclass(frozen=True, slots=True)
class EpochRecord:
    epoch: int
    train_metrics: dict[str, float]
    eval_metrics: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "epoch": self.epoch,
            "train_metrics": self.train_metrics,
        }
        if self.eval_metrics is not None:
            payload["eval_metrics"] = self.eval_metrics
        return payload


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_index: int
    epochs: list[EpochRecord]
    best_epoch: int
    best_metrics: dict[str, float]
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "duration_seconds": self.duration_seconds,
            "epochs": [epoch.to_dict() for epoch in self.epochs],
        }


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    config: TrainerConfig
    runs: list[RunRecord]
    aggregate: dict[str, float]

    @classmethod
    def from_runs(cls, config: TrainerConfig, runs: list[RunRecord]) -> "TrainingSummary":
        best_values = [run.best_metrics[config.selection_metric] for run in runs]
        aggregate = {
            "mean_best_metric": fmean(best_values),
            "std_best_metric": pstdev(best_values) if len(best_values) > 1 else 0.0,
            "best_metric": max(best_values) if config.higher_is_better else min(best_values),
            "worst_metric": min(best_values) if config.higher_is_better else max(best_values),
        }
        return cls(config=config, runs=runs, aggregate=aggregate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trainer_config": {
                "num_epochs": self.config.num_epochs,
                "num_runs": self.config.num_runs,
                "eval_interval": self.config.eval_interval,
                "checkpoint_interval": self.config.checkpoint_interval,
                "selection_metric": self.config.selection_metric,
                "higher_is_better": self.config.higher_is_better,
            },
            "aggregate": self.aggregate,
            "runs": [run.to_dict() for run in self.runs],
        }

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


class TrainerBase(ABC):
    """
    Generic multi-run training loop with best-metric tracking and persisted summaries.

    Concrete subclasses implement backend-specific hooks for one training epoch and one
    evaluation pass. This ports the useful shape of Arch's trainer orchestration without
    forcing the old monolithic PyTorch stack into the new repo.
    """

    def __init__(
        self,
        trainer_config: TrainerConfig,
        *,
        experiment_config: ExperimentConfig | None = None,
        paths: ExperimentPaths | None = None,
    ) -> None:
        self.trainer_config = trainer_config
        self.experiment_config = experiment_config
        self.paths = paths
        self._epoch_callback_every: int | None = None
        self._epoch_callback: EpochCallback | None = None
        self._stop_condition: StopCondition | None = None

    def set_epoch_callback(self, every_n_epochs: int, callback: EpochCallback) -> None:
        if every_n_epochs <= 0:
            raise ValueError("every_n_epochs must be positive")
        self._epoch_callback_every = every_n_epochs
        self._epoch_callback = callback

    def set_stop_condition(self, callback: StopCondition) -> None:
        self._stop_condition = callback

    def setup_run(self, run_index: int) -> None:
        """Optional hook called before each training run."""

    def save_checkpoint(self, run_index: int, epoch: int) -> None:
        """Optional checkpoint hook called at the configured interval."""

    def on_best_checkpoint(self, run_index: int, epoch: int, metrics: dict[str, float]) -> None:
        """Optional hook called whenever a new best evaluation is observed."""

    @abstractmethod
    def train_epoch(self, epoch: int) -> MetricMap:
        """Run one training epoch and return scalar metrics."""

    @abstractmethod
    def evaluate_epoch(self, epoch: int) -> MetricMap:
        """Run evaluation and return scalar metrics."""

    def run(self) -> TrainingSummary:
        if self.paths is not None:
            self.paths.create()
            if self.experiment_config is not None:
                self.paths.initialize(self.experiment_config)

        run_records: list[RunRecord] = []
        for run_index in range(1, self.trainer_config.num_runs + 1):
            self.setup_run(run_index)

            epoch_records: list[EpochRecord] = []
            best_epoch = -1
            best_metrics: dict[str, float] | None = None
            best_value = float("-inf") if self.trainer_config.higher_is_better else float("inf")
            start_time = perf_counter()

            for epoch in range(1, self.trainer_config.num_epochs + 1):
                train_metrics = _coerce_metric_map(self.train_epoch(epoch))
                eval_metrics: dict[str, float] | None = None

                should_eval = (
                    epoch % self.trainer_config.eval_interval == 0
                    or epoch == self.trainer_config.num_epochs
                )
                if should_eval:
                    eval_metrics = _coerce_metric_map(self.evaluate_epoch(epoch))
                    if self.trainer_config.selection_metric not in eval_metrics:
                        raise KeyError(
                            f"selection metric '{self.trainer_config.selection_metric}' "
                            f"missing from eval metrics: {sorted(eval_metrics)}"
                        )

                    current_value = eval_metrics[self.trainer_config.selection_metric]
                    is_better = (
                        current_value > best_value
                        if self.trainer_config.higher_is_better
                        else current_value < best_value
                    )
                    if is_better:
                        best_value = current_value
                        best_epoch = epoch
                        best_metrics = dict(eval_metrics)
                        self.on_best_checkpoint(run_index, epoch, best_metrics)

                epoch_record = EpochRecord(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    eval_metrics=eval_metrics,
                )
                epoch_records.append(epoch_record)

                interval = self._epoch_callback_every
                if (
                    self._epoch_callback is not None
                    and interval is not None
                    and epoch % interval == 0
                ):
                    self._epoch_callback(epoch_record, run_index, epoch)

                checkpoint_interval = self.trainer_config.checkpoint_interval
                if checkpoint_interval is not None and epoch % checkpoint_interval == 0:
                    self.save_checkpoint(run_index, epoch)

                if self._stop_condition is not None and not self._stop_condition(
                    epoch_records,
                    run_index,
                    epoch,
                ):
                    break

            if best_metrics is None or best_epoch < 0:
                raise RuntimeError("training completed without producing any evaluation metrics")

            run_records.append(
                RunRecord(
                    run_index=run_index,
                    epochs=epoch_records,
                    best_epoch=best_epoch,
                    best_metrics=best_metrics,
                    duration_seconds=perf_counter() - start_time,
                )
            )

        summary = TrainingSummary.from_runs(self.trainer_config, run_records)
        if self.paths is not None:
            summary.write_json(self.paths.result_path)
        return summary
