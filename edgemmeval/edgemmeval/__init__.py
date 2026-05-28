"""
EdgeMMEval: Edge MMLLM Benchmark Harness.

Usage:
    from edgemmeval.harness import BenchmarkEngine
    from edgemmeval.tasks.image_classification import ImageClassificationTask
    from edgemmeval import report

    engine = BenchmarkEngine()
    run = engine.run(ImageClassificationTask("/data/imagenette2"))
    report.print_summary([run])
    report.save_json([run], "results.json")
"""
from .harness import BenchmarkEngine, BenchmarkRun
from .task import Task, ModelConfig, ModelResult
from .metrics import ecr, ECRResult
from . import report
from . import memory

__all__ = [
    "BenchmarkEngine",
    "BenchmarkRun",
    "Task",
    "ModelConfig",
    "ModelResult",
    "ecr",
    "ECRResult",
    "report",
    "memory",
]
