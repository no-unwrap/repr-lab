"""Core package for repr-lab."""

from repr_lab.analysis import (
    AmbientDimensionStat,
    ClassGeometryStat,
    EffectiveRankStat,
    LayerStatistic,
    LayerwiseAnalysisResult,
    LayerwiseAnalyzer,
    ParticipationRatioStat,
)
from repr_lab.benchmark_result import BenchmarkResultRecord, BenchmarkRunStatus
from repr_lab.checkpointing import (
    CheckpointSourceKind,
    CheckpointSpec,
    ExecutionMaturity,
    HuggingFaceCLIStatus,
    PreparedCheckpoint,
    ResolvedCheckpoint,
    inspect_huggingface_cli,
    load_checkpoint_manifest,
    prepare_checkpoint,
    prepare_model_checkpoint,
    record_local_checkpoint,
)
from repr_lab.config import ExperimentConfig, config_digest, expand_experiment_grid
from repr_lab.dataset import DatasetSpec
from repr_lab.doctor import CheckStatus, DoctorCheck, DoctorReport, run_doctor
from repr_lab.execution import LinearProbeModel, execute_release_plan
from repr_lab.experiment import ExperimentManifest, ExperimentPaths
from repr_lab.features import FeatureCollection
from repr_lab.materialization import (
    VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_NAME,
    VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_VERSION,
    LocalFeatureExport,
    MaterializedReleaseFeatures,
    OptionalRawMediaRuntimeContract,
    VJEPALocalRawMediaExport,
    materialize_release_features,
    optional_raw_media_runtime_contract,
)
from repr_lab.model_catalog import (
    BenchmarkTaskSpec,
    DatasetInterface,
    InputModality,
    ModelFamilySpec,
    ModelVariantSpec,
    PromptInterface,
    RuntimeRequirement,
    list_benchmark_tasks,
    list_model_families,
    resolve_benchmark_task,
    resolve_model_family,
)
from repr_lab.planning import PlannedReleaseExperiment, plan_release_experiment
from repr_lab.published_dataset import PublishedRelease, load_published_release
from repr_lab.registry import Registry
from repr_lab.reporting import RunSummaryRecord, SummaryWarning, build_run_summary
from repr_lab.run_contract import (
    RUN_DIRECTORY_CONTRACT_NAME,
    RUN_DIRECTORY_CONTRACT_VERSION,
    RUN_DIRECTORY_KIND,
    RUN_DIRECTORY_PRODUCER_REPO,
)
from repr_lab.trainer import EpochRecord, RunRecord, TrainerBase, TrainerConfig, TrainingSummary

__version__ = "0.1.0"

__all__ = [
    "AmbientDimensionStat",
    "BenchmarkResultRecord",
    "BenchmarkRunStatus",
    "BenchmarkTaskSpec",
    "CheckStatus",
    "CheckpointSourceKind",
    "CheckpointSpec",
    "ClassGeometryStat",
    "DatasetInterface",
    "DatasetSpec",
    "DoctorCheck",
    "DoctorReport",
    "EffectiveRankStat",
    "EpochRecord",
    "ExecutionMaturity",
    "ExperimentConfig",
    "ExperimentManifest",
    "ExperimentPaths",
    "FeatureCollection",
    "HuggingFaceCLIStatus",
    "InputModality",
    "LayerStatistic",
    "LayerwiseAnalysisResult",
    "LayerwiseAnalyzer",
    "LinearProbeModel",
    "LocalFeatureExport",
    "MaterializedReleaseFeatures",
    "OptionalRawMediaRuntimeContract",
    "ModelFamilySpec",
    "ModelVariantSpec",
    "ParticipationRatioStat",
    "PlannedReleaseExperiment",
    "PreparedCheckpoint",
    "PromptInterface",
    "PublishedRelease",
    "RUN_DIRECTORY_CONTRACT_NAME",
    "RUN_DIRECTORY_CONTRACT_VERSION",
    "RUN_DIRECTORY_KIND",
    "RUN_DIRECTORY_PRODUCER_REPO",
    "Registry",
    "ResolvedCheckpoint",
    "RunRecord",
    "RunSummaryRecord",
    "RuntimeRequirement",
    "SummaryWarning",
    "TrainerBase",
    "TrainerConfig",
    "TrainingSummary",
    "VJEPALocalRawMediaExport",
    "VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_NAME",
    "VJEPA_LOCAL_RAW_MEDIA_EXPORT_CONTRACT_VERSION",
    "__version__",
    "build_run_summary",
    "config_digest",
    "execute_release_plan",
    "expand_experiment_grid",
    "inspect_huggingface_cli",
    "list_benchmark_tasks",
    "list_model_families",
    "load_checkpoint_manifest",
    "load_published_release",
    "materialize_release_features",
    "optional_raw_media_runtime_contract",
    "plan_release_experiment",
    "prepare_checkpoint",
    "prepare_model_checkpoint",
    "record_local_checkpoint",
    "resolve_benchmark_task",
    "resolve_model_family",
    "run_doctor",
]
