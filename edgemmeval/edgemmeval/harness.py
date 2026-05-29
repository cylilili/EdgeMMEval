"""
BenchmarkEngine: orchestrates the two-axis evaluation (deployability + task-level).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from .task import Task, ModelResult
from .metrics import ecr, accuracy, ECRResult
from . import deployability as dep


@dataclass
class BenchmarkRun:
    task_name: str
    hardware_id: str
    deployability: List[dep.DeployabilityEntry]
    model_results: dict            # model_name -> ModelResult
    ecr_result: Optional[ECRResult] = None


class BenchmarkEngine:
    """
    Runs a Task on the current machine.

    Two-axis evaluation:
      1. Deployability pass: probe each model; record load success, memory, time.
      2. Evaluation pass: for deployable models, run inference on the dataset
         and record accuracy + per-sample latency.
    """

    def __init__(
        self,
        hardware_id: Optional[str] = None,
        probe_deploy: bool = True,
        verbose: bool = True,
    ):
        self.hardware_id = hardware_id or dep.current_hardware_id()
        self.probe_deploy = probe_deploy
        self.verbose = verbose

    def run(self, task: Task, n_samples: Optional[int] = None) -> BenchmarkRun:
        if self.verbose:
            print(f"\n{'='*64}")
            print(f"  Task: {task.name}   Hardware: {self.hardware_id}")
            print(f"{'='*64}")

        all_cfgs = task.classic_models() + task.mmllm_models()

        # ---- 1. Deployability pass ----------------------------------------
        deploy_entries: List[dep.DeployabilityEntry] = []
        deploy_map: dict = {}
        if self.probe_deploy:
            if self.verbose:
                print("\n[1/2] Deployability probe")
            for cfg in all_cfgs:
                entry = dep.probe(cfg, self.hardware_id)
                deploy_entries.append(entry)
                deploy_map[cfg.name] = entry
                if self.verbose:
                    print(f"  {cfg.name:<36} [{cfg.kind:<7}]  {entry.status_str()}")

        # ---- 2. Evaluation pass -------------------------------------------
        if self.verbose:
            print("\n[2/2] Evaluation")

        dataset = task.load_dataset()
        if n_samples is not None:
            dataset = dataset[:n_samples]

        model_results: dict = {}
        for cfg in all_cfgs:
            if self.probe_deploy:
                entry = deploy_map.get(cfg.name)
                if entry and not entry.deployable:
                    model_results[cfg.name] = ModelResult(
                        model_name=cfg.name, kind=cfg.kind,
                        accuracy=0.0, latencies_ms=[],
                        deployable=False, deploy_error=entry.error,
                    )
                    continue

            if self.verbose:
                print(f"  {cfg.name:<36} ...", end=" ", flush=True)

            try:
                model = cfg.loader()
                latencies, scores = [], []
                for inp, gt in dataset:
                    t0 = time.perf_counter()
                    pred = cfg.infer(model, inp)
                    latencies.append((time.perf_counter() - t0) * 1000.0)
                    scores.append(task.score(pred, gt))

                acc = accuracy(scores)
                model_results[cfg.name] = ModelResult(
                    model_name=cfg.name, kind=cfg.kind,
                    accuracy=acc, latencies_ms=latencies, deployable=True,
                )
                if self.verbose:
                    r = model_results[cfg.name]
                    print(
                        f"acc={acc:.3f}  "
                        f"lat p50={r.median_latency_ms:.1f}ms "
                        f"[p10={r.p10_latency_ms:.1f}, p90={r.p90_latency_ms:.1f}]"
                    )

            except Exception as exc:
                model_results[cfg.name] = ModelResult(
                    model_name=cfg.name, kind=cfg.kind,
                    accuracy=0.0, latencies_ms=[],
                    deployable=False, deploy_error=str(exc),
                )
                if self.verbose:
                    print(f"ERROR: {exc}")

        ecr_result = _compute_ecr(task.name, model_results)
        if self.verbose and ecr_result:
            print(f"\n  {ecr_result}")

        return BenchmarkRun(
            task_name=task.name,
            hardware_id=self.hardware_id,
            deployability=deploy_entries,
            model_results=model_results,
            ecr_result=ecr_result,
        )


def _compute_ecr(task_name: str, results: dict) -> Optional[ECRResult]:
    classic = [
        r for r in results.values()
        if r.kind == "classic" and r.deployable and r.latencies_ms
    ]
    mmllm = [
        r for r in results.values()
        if r.kind == "mmllm" and r.deployable and r.latencies_ms
    ]
    if not classic or not mmllm:
        return None

    best_c = max(classic, key=lambda r: r.accuracy)
    best_m = max(mmllm, key=lambda r: r.accuracy)
    delta = best_m.accuracy - best_c.accuracy

    try:
        score = ecr(delta, best_m.median_latency_ms, best_c.median_latency_ms)
    except ValueError:
        return None

    return ECRResult(
        task_name=task_name,
        best_mmllm=best_m.model_name,
        best_classic=best_c.model_name,
        delta_acc=delta,
        lat_mmllm_ms=best_m.median_latency_ms,
        lat_classic_ms=best_c.median_latency_ms,
        value=score,
    )
