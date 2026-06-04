"""
Quick smoke-test: Automatic Speech Recognition on LibriSpeech test-clean.

Requirements: LibriSpeech test-clean data (auto-downloaded by torchvision on first run)
              Vosk models under ~/.cache/vosk/ (see README)
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgemmeval import BenchmarkEngine, report
from edgemmeval.tasks import ASRTask

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="/data/librispeech/test-clean", help="LibriSpeech test-clean root")
parser.add_argument("--n-samples", type=int, default=10)
parser.add_argument("--no-deploy-probe", action="store_true")
parser.add_argument("--output", default=None)
args = parser.parse_args()

task = ASRTask(args.data, max_samples=args.n_samples)
engine = BenchmarkEngine(probe_deploy=not args.no_deploy_probe)
run = engine.run(task, n_samples=args.n_samples)
report.print_summary([run])
if args.output:
    report.save_json([run], args.output)
