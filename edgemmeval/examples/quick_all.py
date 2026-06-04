"""
Quick smoke-test: run all five benchmark tasks with a tiny sample budget.

Requires all datasets and models to be set up. See README for setup instructions.
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgemmeval import BenchmarkEngine, report
from edgemmeval.tasks import (
    ImageClassificationTask, ASRTask, ObjectCountingTask,
    ActionRecognitionTask, HazardDetectionTask,
)

parser = argparse.ArgumentParser()
parser.add_argument("--data-ic",          default="/data/imagenette2")
parser.add_argument("--data-asr",         default="/data/librispeech/test-clean")
parser.add_argument("--data-oc-images",   default="/data/coco/val2017")
parser.add_argument("--data-oc-ann",      default="/data/coco/annotations/instances_val2017.json")
parser.add_argument("--data-ar",          default="/data/kinetics/mini_val")
parser.add_argument("--data-hd",          default="/data/detectium_fire")
parser.add_argument("--n-samples",  type=int, default=5)
parser.add_argument("--no-deploy-probe",  action="store_true")
parser.add_argument("--output",           default=None)
args = parser.parse_args()

tasks = [
    ImageClassificationTask(args.data_ic),
    ASRTask(args.data_asr, max_samples=args.n_samples),
    ObjectCountingTask(args.data_oc_images, args.data_oc_ann, max_samples=args.n_samples),
    ActionRecognitionTask(args.data_ar, max_samples=args.n_samples),
    HazardDetectionTask(args.data_hd, max_videos=args.n_samples),
]

engine = BenchmarkEngine(probe_deploy=not args.no_deploy_probe)
runs = [engine.run(task, n_samples=args.n_samples) for task in tasks]
report.print_summary(runs)
if args.output:
    report.save_json(runs, args.output)
