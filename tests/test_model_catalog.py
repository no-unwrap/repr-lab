from __future__ import annotations

import pytest

from repr_lab import (
    DatasetInterface,
    ExecutionMaturity,
    InputModality,
    PromptInterface,
    list_benchmark_tasks,
    list_model_families,
    resolve_benchmark_task,
    resolve_model_family,
)


def test_seeded_model_families_are_available() -> None:
    families = [family.name for family in list_model_families()]

    assert families == ["sam2", "sam3", "segment-anything", "vjepa2"]


def test_model_family_aliases_resolve() -> None:
    assert resolve_model_family("sam1").name == "segment-anything"
    assert resolve_model_family("sam-2").name == "sam2"
    assert resolve_model_family("segment-anything-3").name == "sam3"
    assert resolve_model_family("v-jepa2").name == "vjepa2"


def test_default_variants_are_stable() -> None:
    assert resolve_model_family("vjepa2").default_variant().name == "vjepa2_1_vit_base_384"
    assert resolve_model_family("sam2").default_variant().name == "sam2.1_hiera_tiny"
    assert resolve_model_family("sam3").default_variant().name == "sam3"
    assert resolve_model_family("segment-anything").default_variant().name == "vit_b"


def test_tasks_can_be_filtered_by_model_family() -> None:
    task_names = [task.name for task in list_benchmark_tasks(model_family="sam2")]

    assert task_names == [
        "automatic-mask-proposal",
        "promptable-mask-refinement",
        "video-mask-propagation",
    ]


def test_sam3_task_catalog_includes_open_vocabulary_tasks() -> None:
    task_names = [task.name for task in list_benchmark_tasks(model_family="sam3")]

    assert task_names == [
        "open-vocabulary-concept-segmentation",
        "open-vocabulary-concept-tracking",
        "promptable-mask-refinement",
        "video-mask-propagation",
    ]


def test_task_aliases_resolve() -> None:
    assert resolve_benchmark_task("mask-refinement").name == "promptable-mask-refinement"
    assert resolve_benchmark_task("temporal-anticipation").name == "action-anticipation"


def test_catalog_tracks_execution_maturity() -> None:
    assert resolve_model_family("vjepa2").maturity is ExecutionMaturity.RUNNABLE
    assert (
        resolve_benchmark_task("frozen-feature-localization-probe").maturity
        is ExecutionMaturity.RUNNABLE
    )
    assert resolve_model_family("sam3").maturity is ExecutionMaturity.PLANNED
    assert (
        resolve_benchmark_task("video-mask-propagation").maturity is ExecutionMaturity.CATALOG_ONLY
    )


def test_model_family_capabilities_are_explicit() -> None:
    sam1 = resolve_model_family("sam1")
    vjepa2 = resolve_model_family("vjepa2")

    assert sam1.supported_modalities == (InputModality.IMAGE,)
    assert PromptInterface.SPATIAL in sam1.prompt_capabilities
    assert vjepa2.supported_modalities == (InputModality.IMAGE, InputModality.VIDEO)
    assert vjepa2.prompt_capabilities == (PromptInterface.NONE,)


def test_task_interface_contracts_are_explicit() -> None:
    image_task = resolve_benchmark_task("open-vocabulary-concept-segmentation")
    video_task = resolve_benchmark_task("video-mask-propagation")

    assert image_task.dataset_interface is DatasetInterface.PUBLISHED_RELEASE
    assert image_task.required_modalities == (InputModality.IMAGE,)
    assert image_task.prompt_interfaces == (PromptInterface.TEXT, PromptInterface.EXEMPLAR)
    assert video_task.dataset_interface is DatasetInterface.VIDEO_SEQUENCE
    assert video_task.required_modalities == (InputModality.VIDEO,)
    assert video_task.prompt_interfaces == (PromptInterface.INITIAL_MASK,)


def test_resolve_model_family_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="Unknown model family"):
        resolve_model_family("nonexistent-model")


def test_resolve_benchmark_task_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="Unknown benchmark task"):
        resolve_benchmark_task("nonexistent-task")


def test_resolve_variant_rejects_unknown_name() -> None:
    family = resolve_model_family("vjepa2")

    with pytest.raises(KeyError, match="Unknown variant"):
        family.resolve_variant("nonexistent-variant")
