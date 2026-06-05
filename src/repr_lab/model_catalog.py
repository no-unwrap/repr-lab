from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from repr_lab.checkpointing import CheckpointSourceKind, CheckpointSpec, ExecutionMaturity
from repr_lab.published_dataset import PublishedRelease
from repr_lab.registry import Registry


def _normalize_identifier(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized


class InputModality(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class PromptInterface(str, Enum):
    NONE = "none"
    SPATIAL = "spatial"
    TEXT = "text"
    EXEMPLAR = "exemplar"
    INITIAL_MASK = "initial_mask"


class DatasetInterface(str, Enum):
    PUBLISHED_RELEASE = "published_release"
    VIDEO_SEQUENCE = "video_sequence"


@dataclass(frozen=True, slots=True)
class RuntimeRequirement:
    python_version: str
    framework: str
    accelerator: str
    install: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "python_version": self.python_version,
            "framework": self.framework,
            "accelerator": self.accelerator,
        }
        if self.install:
            payload["install"] = list(self.install)
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True, slots=True)
class ModelVariantSpec:
    name: str
    display_name: str
    parameter_count: str | None = None
    resolution: int | None = None
    checkpoint: CheckpointSpec | None = None
    recommended: bool = False
    notes: tuple[str, ...] = ()

    def matches(self, value: str) -> bool:
        normalized = _normalize_identifier(value)
        return normalized == _normalize_identifier(self.name)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "display_name": self.display_name,
            "recommended": self.recommended,
        }
        if self.parameter_count is not None:
            payload["parameter_count"] = self.parameter_count
        if self.resolution is not None:
            payload["resolution"] = self.resolution
        if self.checkpoint is not None:
            payload["checkpoint"] = self.checkpoint.to_dict()
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True, slots=True)
class ModelFamilySpec:
    name: str
    display_name: str
    summary: str
    repo_url: str
    integration_role: str
    aliases: tuple[str, ...]
    recommended_tasks: tuple[str, ...]
    runtime: RuntimeRequirement
    supported_modalities: tuple[InputModality, ...]
    prompt_capabilities: tuple[PromptInterface, ...]
    variants: tuple[ModelVariantSpec, ...]
    maturity: ExecutionMaturity = ExecutionMaturity.PLANNED
    notes: tuple[str, ...] = ()

    def matches(self, value: str) -> bool:
        normalized = _normalize_identifier(value)
        candidates = (self.name, *self.aliases)
        return any(_normalize_identifier(candidate) == normalized for candidate in candidates)

    def default_variant(self) -> ModelVariantSpec:
        for variant in self.variants:
            if variant.recommended:
                return variant
        return self.variants[0]

    def resolve_variant(self, value: str | None = None) -> ModelVariantSpec:
        if value is None:
            return self.default_variant()
        normalized = _normalize_identifier(value)
        for variant in self.variants:
            if variant.matches(normalized):
                return variant
        known = ", ".join(variant.name for variant in self.variants)
        raise KeyError(
            f"Unknown variant '{value}' for model family '{self.name}'. Known values: {known}"
        )

    def to_dict(self, *, include_variants: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "display_name": self.display_name,
            "summary": self.summary,
            "repo_url": self.repo_url,
            "integration_role": self.integration_role,
            "aliases": list(self.aliases),
            "recommended_tasks": list(self.recommended_tasks),
            "runtime": self.runtime.to_dict(),
            "supported_modalities": [modality.value for modality in self.supported_modalities],
            "prompt_capabilities": [prompt.value for prompt in self.prompt_capabilities],
            "default_variant": self.default_variant().name,
            "maturity": self.maturity.value,
        }
        if self.notes:
            payload["notes"] = list(self.notes)
        if include_variants:
            payload["variants"] = [variant.to_dict() for variant in self.variants]
        return payload


@dataclass(frozen=True, slots=True)
class BenchmarkTaskSpec:
    name: str
    display_name: str
    summary: str
    compatible_model_families: tuple[str, ...]
    metrics: tuple[str, ...]
    outputs: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    dataset_interface: DatasetInterface = DatasetInterface.PUBLISHED_RELEASE
    required_modalities: tuple[InputModality, ...] = (InputModality.IMAGE,)
    prompt_interfaces: tuple[PromptInterface, ...] = (PromptInterface.NONE,)
    release_bundle_required: bool = True
    required_release_task_types: tuple[str, ...] = ()
    maturity: ExecutionMaturity = ExecutionMaturity.PLANNED
    notes: tuple[str, ...] = ()

    def matches(self, value: str) -> bool:
        normalized = _normalize_identifier(value)
        candidates = (self.name, *self.aliases)
        return any(_normalize_identifier(candidate) == normalized for candidate in candidates)

    def supports_model_family(self, family_name: str) -> bool:
        normalized = _normalize_identifier(family_name)
        return any(
            _normalize_identifier(candidate) == normalized
            for candidate in self.compatible_model_families
        )

    def validate_model_family(self, family: ModelFamilySpec) -> None:
        if not self.supports_model_family(family.name):
            raise ValueError(
                f"Task '{self.name}' is not compatible with model family '{family.name}'"
            )
        family_modalities = {modality.value for modality in family.supported_modalities}
        required_modalities = {modality.value for modality in self.required_modalities}
        if not required_modalities.issubset(family_modalities):
            expected = ", ".join(sorted(required_modalities))
            observed = ", ".join(sorted(family_modalities))
            raise ValueError(
                f"Task '{self.name}' requires modalities [{expected}], got [{observed}]"
            )
        family_prompts = {prompt.value for prompt in family.prompt_capabilities}
        required_prompts = {prompt.value for prompt in self.prompt_interfaces}
        if not family_prompts.intersection(required_prompts):
            expected = ", ".join(sorted(required_prompts))
            observed = ", ".join(sorted(family_prompts))
            raise ValueError(
                f"Task '{self.name}' requires prompt interfaces [{expected}], got [{observed}]"
            )

    def validate_release(self, release: PublishedRelease) -> None:
        if not self.release_bundle_required:
            return
        if not self.required_release_task_types:
            return
        available = {_normalize_identifier(task_type) for task_type in release.task_types}
        required = {
            _normalize_identifier(task_type) for task_type in self.required_release_task_types
        }
        if available.isdisjoint(required):
            expected = ", ".join(sorted(required))
            observed = ", ".join(sorted(available)) or "<empty>"
            raise ValueError(
                f"Task '{self.name}' requires release task types [{expected}], got [{observed}]"
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "display_name": self.display_name,
            "summary": self.summary,
            "compatible_model_families": list(self.compatible_model_families),
            "metrics": list(self.metrics),
            "outputs": list(self.outputs),
            "dataset_interface": self.dataset_interface.value,
            "required_modalities": [modality.value for modality in self.required_modalities],
            "prompt_interfaces": [prompt.value for prompt in self.prompt_interfaces],
            "release_bundle_required": self.release_bundle_required,
            "maturity": self.maturity.value,
        }
        if self.aliases:
            payload["aliases"] = list(self.aliases)
        if self.required_release_task_types:
            payload["required_release_task_types"] = list(self.required_release_task_types)
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


MODEL_FAMILY_REGISTRY = Registry[ModelFamilySpec]("model family")
BENCHMARK_TASK_REGISTRY = Registry[BenchmarkTaskSpec]("benchmark task")


def _register_model_families() -> None:
    families = [
        ModelFamilySpec(
            name="vjepa2",
            display_name="V-JEPA 2.1",
            summary=(
                "Self-supervised video representation family for dense features, "
                "frozen-backbone probes, and temporal anticipation."
            ),
            repo_url="https://github.com/facebookresearch/vjepa2",
            integration_role="offline representation backbone",
            aliases=("v-jepa-2", "v-jepa2", "vjepa-2"),
            recommended_tasks=("frozen-feature-localization-probe", "action-anticipation"),
            runtime=RuntimeRequirement(
                python_version=">=3.12",
                framework="PyTorch Hub or Hugging Face Transformers",
                accelerator="CUDA strongly recommended",
                install=(
                    "pip install torch timm einops",
                    "Optional: use Hugging Face checkpoints or PyTorch Hub loading",
                ),
                notes=(
                    "Official repo recommends CUDA support.",
                    "Official code depends on a decord-compatible video loader; "
                    "macOS needs a replacement implementation.",
                ),
            ),
            supported_modalities=(InputModality.IMAGE, InputModality.VIDEO),
            prompt_capabilities=(PromptInterface.NONE,),
            variants=(
                ModelVariantSpec(
                    name="vjepa2_1_vit_base_384",
                    display_name="V-JEPA 2.1 ViT-B/16 384",
                    parameter_count="80M",
                    resolution=384,
                    checkpoint=CheckpointSpec(
                        source_kind=CheckpointSourceKind.TORCH_HUB,
                        locator="facebookresearch/vjepa2",
                        revision="main",
                        notes=(
                            "Resolve through torch.hub or the upstream Hugging Face collection.",
                        ),
                    ),
                    recommended=True,
                    notes=("Best first bounded benchmark tier in repr-lab.",),
                ),
                ModelVariantSpec(
                    name="vjepa2_1_vit_large_384",
                    display_name="V-JEPA 2.1 ViT-L/16 384",
                    parameter_count="300M",
                    resolution=384,
                    checkpoint=CheckpointSpec(
                        source_kind=CheckpointSourceKind.TORCH_HUB,
                        locator="facebookresearch/vjepa2",
                        revision="main",
                    ),
                ),
                ModelVariantSpec(
                    name="vjepa2_1_vit_giant_384",
                    display_name="V-JEPA 2.1 ViT-g/16 384",
                    parameter_count="1B",
                    resolution=384,
                    checkpoint=CheckpointSpec(
                        source_kind=CheckpointSourceKind.TORCH_HUB,
                        locator="facebookresearch/vjepa2",
                        revision="main",
                    ),
                ),
                ModelVariantSpec(
                    name="vjepa2_1_vit_gigantic_384",
                    display_name="V-JEPA 2.1 ViT-G/16 384",
                    parameter_count="2B",
                    resolution=384,
                    checkpoint=CheckpointSpec(
                        source_kind=CheckpointSourceKind.TORCH_HUB,
                        locator="facebookresearch/vjepa2",
                        revision="main",
                    ),
                    notes=(
                        "Highest-capacity official checkpoint; expensive for "
                        "routine benchmark sweeps.",
                    ),
                ),
            ),
            maturity=ExecutionMaturity.RUNNABLE,
            notes=(
                "Use this family in repr-lab for frozen-feature probes, "
                "temporal anticipation, and dense-video representation studies.",
                "Do not treat it as an on-device XR runtime model.",
                "The first runnable path in repr-lab consumes explicit "
                "local-only frozen feature collections for probe execution.",
            ),
        ),
        ModelFamilySpec(
            name="sam2",
            display_name="SAM 2.1",
            summary=(
                "Promptable image and video segmentation family with "
                "multi-object video propagation and tracking support."
            ),
            repo_url="https://github.com/facebookresearch/sam2",
            integration_role="promptable segmentation and video propagation baseline",
            aliases=("sam-2", "segment-anything-2", "segment-anything-2.1"),
            recommended_tasks=(
                "promptable-mask-refinement",
                "automatic-mask-proposal",
                "video-mask-propagation",
            ),
            runtime=RuntimeRequirement(
                python_version=">=3.10",
                framework="PyTorch package",
                accelerator="CUDA recommended; custom CUDA extension during install",
                install=(
                    "git clone https://github.com/facebookresearch/sam2.git && pip install -e .",
                    "pip install -e '.[notebooks]' for notebook examples",
                ),
                notes=(
                    "Official repo requires torch>=2.5.1 and torchvision>=0.20.1.",
                    "nvcc / matching CUDA toolkit is recommended for the custom extension.",
                ),
            ),
            supported_modalities=(InputModality.IMAGE, InputModality.VIDEO),
            prompt_capabilities=(
                PromptInterface.NONE,
                PromptInterface.SPATIAL,
                PromptInterface.INITIAL_MASK,
            ),
            variants=(
                ModelVariantSpec(
                    name="sam2.1_hiera_tiny",
                    display_name="SAM 2.1 Hiera Tiny",
                    checkpoint=CheckpointSpec(
                        source_kind=CheckpointSourceKind.REPO_RELEASE,
                        locator="https://github.com/facebookresearch/sam2",
                        notes=("Use the upstream checkpoint download flow from the sam2 repo.",),
                    ),
                    recommended=True,
                ),
                ModelVariantSpec(
                    name="sam2.1_hiera_small",
                    display_name="SAM 2.1 Hiera Small",
                    checkpoint=CheckpointSpec(
                        source_kind=CheckpointSourceKind.REPO_RELEASE,
                        locator="https://github.com/facebookresearch/sam2",
                    ),
                ),
                ModelVariantSpec(
                    name="sam2.1_hiera_base_plus",
                    display_name="SAM 2.1 Hiera Base+",
                    checkpoint=CheckpointSpec(
                        source_kind=CheckpointSourceKind.REPO_RELEASE,
                        locator="https://github.com/facebookresearch/sam2",
                    ),
                ),
                ModelVariantSpec(
                    name="sam2.1_hiera_large",
                    display_name="SAM 2.1 Hiera Large",
                    checkpoint=CheckpointSpec(
                        source_kind=CheckpointSourceKind.REPO_RELEASE,
                        locator="https://github.com/facebookresearch/sam2",
                    ),
                    notes=("Best-quality official checkpoint; higher-cost benchmark tier.",),
                ),
            ),
            maturity=ExecutionMaturity.PLANNED,
            notes=(
                "Use SAM 2 when repr-lab needs promptable segmentation on images "
                "plus video mask propagation or tracking.",
                "This family supersedes SAM 1 for video-centric evaluation.",
            ),
        ),
        ModelFamilySpec(
            name="sam3",
            display_name="SAM 3",
            summary=(
                "Unified open-vocabulary image/video segmentation family with "
                "text and visual prompts plus concept tracking."
            ),
            repo_url="https://github.com/facebookresearch/sam3",
            integration_role="open-vocabulary concept segmentation and tracking baseline",
            aliases=("sam-3", "segment-anything-3", "segment-anything-with-concepts"),
            recommended_tasks=(
                "open-vocabulary-concept-segmentation",
                "open-vocabulary-concept-tracking",
                "promptable-mask-refinement",
            ),
            runtime=RuntimeRequirement(
                python_version=">=3.12",
                framework="PyTorch package plus gated Hugging Face checkpoints",
                accelerator="CUDA-compatible GPU with CUDA 12.6+",
                install=(
                    "git clone https://github.com/facebookresearch/sam3.git && pip install -e .",
                    "Use the authenticated Hugging Face CLI to access gated "
                    "checkpoints from facebook/sam3",
                ),
                notes=(
                    "Official repo requires torch>=2.7.",
                    "Checkpoint access is gated on Hugging Face.",
                ),
            ),
            supported_modalities=(InputModality.IMAGE, InputModality.VIDEO),
            prompt_capabilities=(
                PromptInterface.SPATIAL,
                PromptInterface.TEXT,
                PromptInterface.EXEMPLAR,
                PromptInterface.INITIAL_MASK,
            ),
            variants=(
                ModelVariantSpec(
                    name="sam3",
                    display_name="SAM 3 Unified Model",
                    parameter_count="848M",
                    checkpoint=CheckpointSpec(
                        source_kind=CheckpointSourceKind.HUGGINGFACE,
                        locator="facebook/sam3",
                        gated=True,
                        notes=(
                            "Use the authenticated Hugging Face CLI on this machine "
                            "to materialize the checkpoint.",
                        ),
                    ),
                    recommended=True,
                    notes=(
                        "Text-prompted open-vocabulary concept segmentation and tracking family.",
                    ),
                ),
            ),
            maturity=ExecutionMaturity.PLANNED,
            notes=(
                "Use SAM 3 when repr-lab needs open-vocabulary concept "
                "segmentation or concept tracking on images/videos.",
                "This family is heavier and more operationally constrained "
                "than SAM 1/2 because checkpoint access is gated.",
            ),
        ),
        ModelFamilySpec(
            name="segment-anything",
            display_name="Segment Anything (SAM 1)",
            summary=(
                "Promptable image segmentation family with automatic mask "
                "generation and ONNX-exportable decoder."
            ),
            repo_url="https://github.com/facebookresearch/segment-anything",
            integration_role="static-image promptable segmentation baseline",
            aliases=("sam", "sam1", "segment-anything-model"),
            recommended_tasks=("promptable-mask-refinement", "automatic-mask-proposal"),
            runtime=RuntimeRequirement(
                python_version=">=3.8",
                framework="PyTorch package with optional ONNX export",
                accelerator="CUDA strongly recommended",
                install=(
                    "pip install git+https://github.com/facebookresearch/segment-anything.git",
                    "Optional extras: opencv-python pycocotools matplotlib onnxruntime onnx",
                ),
                notes=(
                    "Official repo supports prompt-based prediction and "
                    "automatic mask generation on static images.",
                    "The lightweight mask decoder can be exported to ONNX for "
                    "portable inference surfaces.",
                ),
            ),
            supported_modalities=(InputModality.IMAGE,),
            prompt_capabilities=(PromptInterface.NONE, PromptInterface.SPATIAL),
            variants=(
                ModelVariantSpec(
                    name="vit_b",
                    display_name="SAM ViT-B",
                    checkpoint=CheckpointSpec(
                        source_kind=CheckpointSourceKind.REPO_RELEASE,
                        locator="https://github.com/facebookresearch/segment-anything",
                        notes=("Use the checkpoint links documented in the upstream repo.",),
                    ),
                    recommended=True,
                    notes=("Best first static-image baseline tier in repr-lab.",),
                ),
                ModelVariantSpec(
                    name="vit_l",
                    display_name="SAM ViT-L",
                    checkpoint=CheckpointSpec(
                        source_kind=CheckpointSourceKind.REPO_RELEASE,
                        locator="https://github.com/facebookresearch/segment-anything",
                    ),
                ),
                ModelVariantSpec(
                    name="vit_h",
                    display_name="SAM ViT-H",
                    checkpoint=CheckpointSpec(
                        source_kind=CheckpointSourceKind.REPO_RELEASE,
                        locator="https://github.com/facebookresearch/segment-anything",
                    ),
                ),
            ),
            maturity=ExecutionMaturity.PLANNED,
            notes=(
                "Use SAM 1 as the lighter static-image promptable baseline and "
                "ONNX-friendly export surface.",
                "Prefer SAM 2 for video tasks or when one codepath should span images and videos.",
            ),
        ),
    ]

    for family in families:
        MODEL_FAMILY_REGISTRY.register(family.name, family)


def _register_benchmark_tasks() -> None:
    tasks = [
        BenchmarkTaskSpec(
            name="frozen-feature-localization-probe",
            display_name="Frozen Feature Localization Probe",
            summary=(
                "Train a lightweight probe over frozen backbone features from "
                "a published release to predict localization-relevant labels."
            ),
            compatible_model_families=("vjepa2",),
            metrics=("mean_average_precision", "macro_f1", "top1_accuracy"),
            outputs=("probe_metrics", "feature_store", "confusion_summary"),
            aliases=("localization-probe", "feature-probe"),
            required_modalities=(InputModality.IMAGE,),
            prompt_interfaces=(PromptInterface.NONE,),
            required_release_task_types=("bbox", "polygon", "segmentation", "mask"),
            maturity=ExecutionMaturity.RUNNABLE,
            notes=(
                "Primary repr-lab entry point for V-JEPA 2.1 on published "
                "static-image releases.",
                "Current runnable path expects explicit local feature "
                "collections with plan-matched provenance metadata.",
            ),
        ),
        BenchmarkTaskSpec(
            name="automatic-mask-proposal",
            display_name="Automatic Mask Proposal",
            summary=(
                "Generate class-agnostic mask proposals over a published "
                "release and score coverage against annotations."
            ),
            compatible_model_families=("segment-anything", "sam2"),
            metrics=("proposal_recall_at_10", "mean_iou", "annotation_coverage"),
            outputs=("mask_proposals", "proposal_metrics", "coverage_report"),
            aliases=("amg",),
            required_modalities=(InputModality.IMAGE,),
            prompt_interfaces=(PromptInterface.NONE,),
            required_release_task_types=("bbox", "polygon", "segmentation", "mask"),
            maturity=ExecutionMaturity.PLANNED,
        ),
        BenchmarkTaskSpec(
            name="promptable-mask-refinement",
            display_name="Promptable Mask Refinement",
            summary=(
                "Use annotation-derived prompts to refine object masks and "
                "benchmark promptable segmentation quality."
            ),
            compatible_model_families=("segment-anything", "sam2", "sam3"),
            metrics=("mean_iou", "boundary_f1", "prompt_success_rate"),
            outputs=("prompted_masks", "mask_metrics", "prompt_audit"),
            aliases=("mask-refinement", "promptable-segmentation"),
            required_modalities=(InputModality.IMAGE,),
            prompt_interfaces=(PromptInterface.SPATIAL,),
            required_release_task_types=("bbox", "polygon", "segmentation", "mask"),
            maturity=ExecutionMaturity.PLANNED,
        ),
        BenchmarkTaskSpec(
            name="open-vocabulary-concept-segmentation",
            display_name="Open-Vocabulary Concept Segmentation",
            summary=(
                "Use text or exemplar prompts to segment all instances of a "
                "concept in a published release."
            ),
            compatible_model_families=("sam3",),
            metrics=("concept_group_f1", "mean_iou", "open_vocab_recall"),
            outputs=("concept_masks", "concept_metrics", "prompt_audit"),
            aliases=("concept-segmentation", "pcs"),
            required_modalities=(InputModality.IMAGE,),
            prompt_interfaces=(PromptInterface.TEXT, PromptInterface.EXEMPLAR),
            required_release_task_types=("bbox", "polygon", "segmentation", "mask"),
            maturity=ExecutionMaturity.PLANNED,
            notes=("Primary SAM 3 image benchmark in repr-lab.",),
        ),
        BenchmarkTaskSpec(
            name="video-mask-propagation",
            display_name="Video Mask Propagation",
            summary=(
                "Propagate prompted masks through a video sequence and "
                "benchmark object tracking consistency."
            ),
            compatible_model_families=("sam2", "sam3"),
            metrics=("vos_j_score", "vos_f_score", "track_fragmentation"),
            outputs=("propagated_masks", "video_metrics", "tracking_audit"),
            aliases=("vos", "video-segmentation"),
            dataset_interface=DatasetInterface.VIDEO_SEQUENCE,
            required_modalities=(InputModality.VIDEO,),
            prompt_interfaces=(PromptInterface.INITIAL_MASK,),
            release_bundle_required=False,
            maturity=ExecutionMaturity.CATALOG_ONLY,
            notes=("Not currently compatible with the static published release bundle contract.",),
        ),
        BenchmarkTaskSpec(
            name="open-vocabulary-concept-tracking",
            display_name="Open-Vocabulary Concept Tracking",
            summary="Track all instances matching a text concept through a video sequence.",
            compatible_model_families=("sam3",),
            metrics=("concept_track_f1", "hota", "track_fragmentation"),
            outputs=("tracked_concepts", "tracking_metrics", "session_audit"),
            aliases=("concept-tracking",),
            dataset_interface=DatasetInterface.VIDEO_SEQUENCE,
            required_modalities=(InputModality.VIDEO,),
            prompt_interfaces=(PromptInterface.TEXT, PromptInterface.EXEMPLAR),
            release_bundle_required=False,
            maturity=ExecutionMaturity.CATALOG_ONLY,
            notes=("Primary SAM 3 video benchmark in repr-lab.",),
        ),
        BenchmarkTaskSpec(
            name="action-anticipation",
            display_name="Action Anticipation",
            summary=(
                "Use a video representation backbone for temporal anticipation "
                "and downstream evaluators."
            ),
            compatible_model_families=("vjepa2",),
            metrics=("recall_at_5", "top1_accuracy", "latency_ms"),
            outputs=("anticipation_metrics", "temporal_features", "evaluator_summary"),
            aliases=("temporal-anticipation",),
            dataset_interface=DatasetInterface.VIDEO_SEQUENCE,
            required_modalities=(InputModality.VIDEO,),
            prompt_interfaces=(PromptInterface.NONE,),
            release_bundle_required=False,
            maturity=ExecutionMaturity.CATALOG_ONLY,
            notes=(
                "Intended for future video-native dataset adapters rather than "
                "the current static release reader.",
            ),
        ),
    ]

    for task in tasks:
        BENCHMARK_TASK_REGISTRY.register(task.name, task)


_register_model_families()
_register_benchmark_tasks()


def list_model_families() -> list[ModelFamilySpec]:
    return [family for _, family in MODEL_FAMILY_REGISTRY.items()]


def list_benchmark_tasks(*, model_family: str | None = None) -> list[BenchmarkTaskSpec]:
    tasks = [task for _, task in BENCHMARK_TASK_REGISTRY.items()]
    if model_family is None:
        return tasks
    family = resolve_model_family(model_family)
    return [task for task in tasks if task.supports_model_family(family.name)]


def resolve_model_family(name: str) -> ModelFamilySpec:
    normalized = _normalize_identifier(name)
    for family in list_model_families():
        if family.matches(normalized):
            return family
    known = ", ".join(family.name for family in list_model_families())
    raise KeyError(f"Unknown model family '{name}'. Known values: {known}")


def resolve_benchmark_task(name: str) -> BenchmarkTaskSpec:
    normalized = _normalize_identifier(name)
    for task in list_benchmark_tasks():
        if task.matches(normalized):
            return task
    known = ", ".join(task.name for task in list_benchmark_tasks())
    raise KeyError(f"Unknown benchmark task '{name}'. Known values: {known}")
