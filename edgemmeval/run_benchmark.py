#!/usr/bin/env python3
"""
Edge MMLLM Benchmark — CLI entry point.

Examples
--------
# Run all tasks with default data paths:
  python run_benchmark.py --all

# Run a single task:
  python run_benchmark.py --task ic --data-ic /data/imagenette2

# Quick smoke-test (5 samples per task):
  python run_benchmark.py --all --n-samples 5

# Skip deployability probe (faster iteration):
  python run_benchmark.py --task ic --data-ic /data/imagenette2 --no-deploy-probe

# Save results to JSON:
  python run_benchmark.py --all --output results.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from edgemmeval import BenchmarkEngine, report
from edgemmeval.tasks import (
    ActionRecognitionTask,
    ASRTask,
    HazardDetectionTask,
    ImageClassificationTask,
    ObjectCountingTask,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ecosystem-aware benchmark: MMLLMs vs. specialized edge models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task selection
    task_group = p.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--all", action="store_true", help="Run all five tasks")
    task_group.add_argument(
        "--task",
        choices=["ic", "asr", "oc", "ar", "hd"],
        help="Run a single task: ic=image classification, asr=speech recognition, "
             "oc=object counting, ar=action recognition, hd=hazard detection",
    )

    # Dataset paths
    p.add_argument("--data-ic",  default="/data/imagenette2",       metavar="PATH", help="Imagenette root dir")
    p.add_argument("--data-asr", default="/data/librispeech/test-clean", metavar="PATH", help="LibriSpeech test-clean root")
    p.add_argument("--data-oc-images",  default="/data/coco/val2017",    metavar="PATH", help="COCO val2017 images")
    p.add_argument("--data-oc-ann",     default="/data/coco/annotations/instances_val2017.json", metavar="PATH", help="COCO instances annotation JSON")
    p.add_argument("--data-ar", default="/data/kinetics/mini_val",   metavar="PATH", help="Kinetics mini-val root")
    p.add_argument("--data-hd", default="/data/detectium_fire",      metavar="PATH", help="DetectiumFire dataset root")

    # Task-specific options
    p.add_argument("--oc-class", default="person",  help="COCO class to count (object counting task)")
    p.add_argument("--hd-k",    type=int, default=5, help="Periodic invocation interval k (hazard detection)")

    # Engine options — sample budget
    budget_group = p.add_mutually_exclusive_group()
    budget_group.add_argument(
        "--budget",
        choices=["quick", "standard", "full"],
        default=None,
        help=(
            "Preset sample budget: "
            "quick=50 samples (smoke-test, ~mins), "
            "standard=500 samples (reasonable estimate, ~1hr), "
            "full=all samples (publication quality)"
        ),
    )
    budget_group.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Exact sample cap; overrides --budget",
    )
    p.add_argument("--no-deploy-probe", action="store_true",    help="Skip deployability probe pass")
    p.add_argument("--hardware-id",     default=None,           help="Override hardware label in reports")

    # Output
    p.add_argument("--output", default=None, metavar="FILE.json", help="Save results to JSON file")
    p.add_argument("--quiet",  action="store_true", help="Suppress per-model progress output")

    return p


def build_tasks(args) -> list:
    tasks_to_run = []

    def want(key: str) -> bool:
        return args.all or args.task == key

    if want("ic"):
        tasks_to_run.append(ImageClassificationTask(args.data_ic))

    if want("asr"):
        tasks_to_run.append(ASRTask(args.data_asr, max_samples=args.n_samples or 500))

    if want("oc"):
        tasks_to_run.append(ObjectCountingTask(
            images_dir=args.data_oc_images,
            annotations_file=args.data_oc_ann,
            target_class=args.oc_class,
            max_samples=args.n_samples or 500,
        ))

    if want("ar"):
        tasks_to_run.append(ActionRecognitionTask(args.data_ar, max_samples=args.n_samples or 50))

    if want("hd"):
        tasks_to_run.append(HazardDetectionTask(
            args.data_hd,
            max_videos=args.n_samples or 200,
            k_periodic=args.hd_k,
        ))

    return tasks_to_run


_BUDGET_SAMPLES = {"quick": 50, "standard": 500, "full": None}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Resolve sample cap: --n-samples takes priority; --budget is a preset
    if args.n_samples is None and args.budget is not None:
        args.n_samples = _BUDGET_SAMPLES[args.budget]

    tasks = build_tasks(args)
    if not tasks:
        print("No tasks selected. Use --all or --task <id>.", file=sys.stderr)
        sys.exit(1)

    engine = BenchmarkEngine(
        hardware_id=args.hardware_id,
        probe_deploy=not args.no_deploy_probe,
        verbose=not args.quiet,
    )

    runs = []
    for task in tasks:
        run = engine.run(task, n_samples=args.n_samples)
        runs.append(run)

    report.print_summary(runs)

    if args.output:
        report.save_json(runs, args.output)


if __name__ == "__main__":
    main()
