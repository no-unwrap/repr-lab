from __future__ import annotations

import argparse
import json
from typing import Sequence

from repr_lab import __version__
from repr_lab.checkpointing import prepare_model_checkpoint, record_local_checkpoint
from repr_lab.doctor import run_doctor
from repr_lab.execution import execute_release_plan
from repr_lab.materialization import VJEPALocalRawMediaExport, materialize_release_features
from repr_lab.model_catalog import (
    list_benchmark_tasks,
    list_model_families,
    resolve_benchmark_task,
    resolve_model_family,
)
from repr_lab.planning import plan_release_experiment
from repr_lab.published_dataset import load_published_release


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repr-lab",
        description="Representation-learning experiment framework for published datasets.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the repr-lab package version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")
    inspect_release = subparsers.add_parser(
        "inspect-release",
        help="Validate and summarize a published release bundle.",
    )
    inspect_release.add_argument(
        "--bundle-dir",
        required=True,
        help="Path to the published release bundle directory.",
    )
    list_models = subparsers.add_parser(
        "list-models",
        help="List the benchmark model families currently tracked by repr-lab.",
    )
    list_models.add_argument(
        "--model",
        help="Optional model family filter to return a single resolved family.",
    )
    describe_model = subparsers.add_parser(
        "describe-model",
        help="Describe one tracked benchmark model family.",
    )
    describe_model.add_argument(
        "--model",
        required=True,
        help="Model family name or alias.",
    )
    list_tasks = subparsers.add_parser(
        "list-tasks",
        help="List benchmark tasks known to repr-lab.",
    )
    list_tasks.add_argument(
        "--model",
        help="Optional model family filter.",
    )
    describe_task = subparsers.add_parser(
        "describe-task",
        help="Describe one tracked benchmark task.",
    )
    describe_task.add_argument(
        "--task",
        required=True,
        help="Benchmark task name or alias.",
    )
    plan_experiment = subparsers.add_parser(
        "plan-release-experiment",
        help=(
            "Create an experiment-ready plan for a published release, "
            "model family, and benchmark task."
        ),
    )
    plan_experiment.add_argument(
        "--bundle-dir",
        required=True,
        help="Path to the published release bundle directory.",
    )
    plan_experiment.add_argument(
        "--model",
        required=True,
        help="Model family name or alias.",
    )
    plan_experiment.add_argument(
        "--task",
        required=True,
        help="Benchmark task name or alias.",
    )
    plan_experiment.add_argument(
        "--variant",
        help="Optional model variant. The recommended variant is used when omitted.",
    )
    plan_experiment.add_argument(
        "--experiment-name",
        help="Optional experiment name override.",
    )
    plan_experiment.add_argument(
        "--artifacts-root",
        default="artifacts/planned",
        help="Root directory for planned experiment artifacts.",
    )
    plan_experiment.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic experiment seed for the planned run.",
    )
    execute_plan = subparsers.add_parser(
        "execute-release-plan",
        help=(
            "Execute one runnable planned benchmark path from an existing plan "
            "directory using explicit local feature inputs."
        ),
    )
    execute_plan.add_argument(
        "--plan",
        required=True,
        help="Path to a planned run directory or its plan.json file.",
    )
    execute_plan.add_argument(
        "--train-features",
        help=(
            "Optional path to the train-split feature collection (.npz). When omitted, "
            "repr-lab expects canonical materialized train/test features inside the "
            "planned run directory."
        ),
    )
    execute_plan.add_argument(
        "--test-features",
        help=(
            "Optional path to the test-split feature collection (.npz). When omitted, "
            "repr-lab expects canonical materialized train/test features inside the "
            "planned run directory."
        ),
    )
    execute_plan.add_argument(
        "--layer",
        help="Optional shared feature layer to use when multiple layers are present.",
    )
    execute_plan.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength for the linear frozen-feature probe.",
    )
    record_checkpoint = subparsers.add_parser(
        "record-local-checkpoint",
        help=(
            "Write a local-only checkpoint manifest for a tracked model family without "
            "claiming automated download support."
        ),
    )
    record_checkpoint.add_argument(
        "--model",
        required=True,
        help="Model family name or alias.",
    )
    record_checkpoint.add_argument(
        "--variant",
        help="Optional model variant. The recommended variant is used when omitted.",
    )
    record_checkpoint.add_argument(
        "--local-path",
        required=True,
        help="Local file or directory holding the resolved checkpoint assets.",
    )
    record_checkpoint.add_argument(
        "--output",
        required=True,
        help="Path to write the normalized checkpoint manifest JSON.",
    )
    record_checkpoint.add_argument(
        "--revision",
        help="Optional resolved revision override for the local checkpoint record.",
    )
    record_checkpoint.add_argument(
        "--file",
        action="append",
        default=[],
        help="Optional local file path relative to --local-path. Repeatable.",
    )
    record_checkpoint.add_argument(
        "--note",
        action="append",
        default=[],
        help="Optional local-only checkpoint note. Repeatable.",
    )
    materialize_features = subparsers.add_parser(
        "materialize-release-features",
        help=(
            "Turn a local V-JEPA export plus checkpoint manifest into canonical "
            "train/test FeatureCollection artifacts inside a planned run directory."
        ),
    )
    materialize_features.add_argument(
        "--plan",
        required=True,
        help="Path to a planned run directory or its plan.json file.",
    )
    materialize_features.add_argument(
        "--feature-export",
        required=True,
        help="Path to the local feature-export batch (.npz).",
    )
    materialize_features.add_argument(
        "--checkpoint-manifest",
        required=True,
        help="Path to a local-only or prepared checkpoint manifest JSON.",
    )
    inspect_vjepa_export = subparsers.add_parser(
        "inspect-vjepa-export",
        help="Validate and summarize a bounded local V-JEPA raw-media export batch.",
    )
    inspect_vjepa_export.add_argument(
        "--feature-export",
        required=True,
        help="Path to the local V-JEPA raw-media export batch (.npz).",
    )
    prepare_checkpoint = subparsers.add_parser(
        "prepare-checkpoint",
        help="Prepare checkpoint assets for a tracked model family when automation is supported.",
    )
    prepare_checkpoint.add_argument(
        "--model",
        required=True,
        help="Model family name or alias.",
    )
    prepare_checkpoint.add_argument(
        "--variant",
        help="Optional model variant. The recommended variant is used when omitted.",
    )
    prepare_checkpoint.add_argument(
        "--output-root",
        default="artifacts/checkpoints",
        help="Root directory for prepared checkpoint assets.",
    )
    prepare_checkpoint.add_argument(
        "--revision",
        help="Optional checkpoint revision override.",
    )
    prepare_checkpoint.add_argument(
        "--filename",
        action="append",
        default=[],
        help="Optional filename to download from a Hugging Face repo. Repeatable.",
    )
    prepare_checkpoint.add_argument(
        "--include",
        action="append",
        default=[],
        help="Optional Hugging Face include pattern. Repeatable.",
    )
    prepare_checkpoint.add_argument(
        "--force",
        action="store_true",
        help="Force checkpoint re-download when the automation backend supports it.",
    )
    subparsers.add_parser(
        "doctor",
        help="Report benchmark-suite environment readiness and automation blockers.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"repr-lab {__version__}")
        return 0

    if args.command == "inspect-release":
        release = load_published_release(args.bundle_dir)
        print(json.dumps(release.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "list-models":
        payload: object
        if args.model:
            payload = resolve_model_family(args.model).to_dict()
        else:
            payload = [family.to_dict(include_variants=False) for family in list_model_families()]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "describe-model":
        payload = resolve_model_family(args.model).to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "list-tasks":
        payload = [task.to_dict() for task in list_benchmark_tasks(model_family=args.model)]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "describe-task":
        payload = resolve_benchmark_task(args.task).to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "plan-release-experiment":
        plan = plan_release_experiment(
            args.bundle_dir,
            model_family=args.model,
            task=args.task,
            variant=args.variant,
            experiment_name=args.experiment_name,
            artifacts_root=args.artifacts_root,
            seed=args.seed,
        ).write()
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "execute-release-plan":
        record = execute_release_plan(
            args.plan,
            train_features=args.train_features,
            test_features=args.test_features,
            layer=args.layer,
            ridge_alpha=args.ridge_alpha,
        )
        print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "record-local-checkpoint":
        manifest = record_local_checkpoint(
            args.model,
            variant=args.variant,
            local_path=args.local_path,
            output_path=args.output,
            revision=args.revision,
            files=tuple(args.file),
            notes=tuple(args.note),
        )
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "materialize-release-features":
        materialized = materialize_release_features(
            args.plan,
            feature_export=args.feature_export,
            checkpoint_manifest=args.checkpoint_manifest,
        )
        print(json.dumps(materialized.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "inspect-vjepa-export":
        export = VJEPALocalRawMediaExport.load(args.feature_export)
        print(json.dumps(export.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "prepare-checkpoint":
        prepared = prepare_model_checkpoint(
            args.model,
            variant=args.variant,
            output_root=args.output_root,
            revision=args.revision,
            filenames=tuple(args.filename),
            include_patterns=tuple(args.include),
            force=args.force,
        )
        print(json.dumps(prepared.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "doctor":
        print(json.dumps(run_doctor().to_dict(), indent=2, sort_keys=True))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
