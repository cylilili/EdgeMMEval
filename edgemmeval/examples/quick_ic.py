"""
Quick smoke-test: Image Classification on Imagenette val split.

Requirements: download Imagenette from https://github.com/fastai/imagenette

Expected runtime: ~1–2 min with 10 samples on CPU.
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgemmeval import BenchmarkEngine, report
from edgemmeval.tasks import ImageClassificationTask

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="/data/imagenette2", help="Path to imagenette2 root")
parser.add_argument("--n-samples", type=int, default=10, help="Number of samples to evaluate")
parser.add_argument("--no-deploy-probe", action="store_true")
parser.add_argument("--output", default=None)
args = parser.parse_args()

task = ImageClassificationTask(args.data)
engine = BenchmarkEngine(probe_deploy=not args.no_deploy_probe)
run = engine.run(task, n_samples=args.n_samples)
report.print_summary([run])
if args.output:
    report.save_json([run], args.output)
