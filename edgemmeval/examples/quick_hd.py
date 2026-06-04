"""
Quick smoke-test: Hazard Detection on DetectiumFire dataset.

Requirements: DetectiumFire dataset (see benchmarking/5_hazard_detection/download_detectiumfire_dataset.py)
              Ollama with gemma3n:e2b pulled
              YOLOv8 fire weights at benchmarking/5_hazard_detection/src/weights/best.pt
Usage: python examples/quick_hd.py --data /path/to/detectium_fire
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgemmeval import BenchmarkEngine, report
from edgemmeval.tasks import HazardDetectionTask

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="/data/detectium_fire", help="DetectiumFire dataset root")
parser.add_argument("--n-samples", type=int, default=10)
parser.add_argument("--k", type=int, default=5, help="Periodic invocation interval k")
parser.add_argument("--no-deploy-probe", action="store_true")
parser.add_argument("--output", default=None)
args = parser.parse_args()

task = HazardDetectionTask(args.data, max_videos=args.n_samples, k_periodic=args.k)
engine = BenchmarkEngine(probe_deploy=not args.no_deploy_probe)
run = engine.run(task, n_samples=args.n_samples)
report.print_summary([run])
if args.output:
    report.save_json([run], args.output)
