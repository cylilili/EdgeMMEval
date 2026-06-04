"""
Report generation: console summary + JSON output.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

from .harness import BenchmarkRun


def print_summary(runs: List[BenchmarkRun]) -> None:
    print("\n" + "=" * 72)
    print("  BENCHMARK SUMMARY")
    print("=" * 72)
    _deployability_matrix(runs)
    _ecr_table(runs)
    _per_task_detail(runs)


def _deployability_matrix(runs: List[BenchmarkRun]) -> None:
    print("\n--- Deployability Matrix ---")
    hw_ids = list(dict.fromkeys(r.hardware_id for r in runs))
    all_entries: dict = {}
    for run in runs:
        for e in run.deployability:
            all_entries.setdefault(e.model_name, {})
            all_entries[e.model_name]["kind"] = e.kind
            all_entries[e.model_name][run.hardware_id] = e.deployable

    col = 14
    header = f"  {'Model':<36}  {'Kind':<7}  " + "  ".join(f"{h[:col]:<{col}}" for h in hw_ids)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for model, info in sorted(all_entries.items()):
        kind = info.get("kind", "")
        row = f"  {model:<36}  {kind:<7}  "
        for hw in hw_ids:
            v = info.get(hw)
            cell = "OK  " if v is True else ("FAIL" if v is False else "?   ")
            row += f"  {cell:<{col}}"
        print(row)


def _ecr_table(runs: List[BenchmarkRun]) -> None:
    print("\n--- ECR Scores ---")
    print(f"  {'Task':<22}  {'ΔAcc':>7}  {'L_MMLLM':>10}  {'L_classic':>10}  {'ECR':>8}  {'Verdict'}")
    print("  " + "-" * 74)
    for run in runs:
        r = run.ecr_result
        if r is None:
            print(f"  {run.task_name:<22}  (no result)")
            continue
        verdict = _verdict(r.value)
        ecr_str = f"{r.value:+.3f}" if math.isfinite(r.value) else f"{r.value}"
        print(
            f"  {r.task_name:<22}  {r.delta_acc:+7.3f}  "
            f"{r.lat_mmllm_ms:>8.1f}ms  {r.lat_classic_ms:>8.1f}ms  "
            f"{ecr_str:>8}  {verdict}"
        )


def _verdict(ecr_val: float) -> str:
    if not math.isfinite(ecr_val):
        return "N/A"
    if ecr_val > 0.1:
        return "MMLLM justified"
    if ecr_val > 0.0:
        return "marginal"
    return "classic preferred"


def _per_task_detail(runs: List[BenchmarkRun]) -> None:
    print("\n--- Per-task Model Detail ---")
    for run in runs:
        print(f"\n  [{run.task_name}]")
        print(f"  {'Model':<36}  {'Kind':<7}  {'Acc':>6}  {'p50 lat':>10}  {'p90 lat':>10}")
        print("  " + "-" * 76)
        for name, r in sorted(run.model_results.items(), key=lambda x: -x[1].accuracy):
            if r.deployable and r.latencies_ms:
                print(
                    f"  {name:<36}  {r.kind:<7}  {r.accuracy:>6.3f}  "
                    f"{r.median_latency_ms:>8.1f}ms  {r.p90_latency_ms:>8.1f}ms"
                )
            else:
                print(f"  {name:<36}  {r.kind:<7}  FAILED ({r.deploy_error[:40]})")


def save_json(runs: List[BenchmarkRun], path: str) -> None:
    data = []
    for run in runs:
        entry = {
            "task": run.task_name,
            "hardware": run.hardware_id,
            "deployability": [
                {
                    "model": e.model_name,
                    "kind": e.kind,
                    "deployable": e.deployable,
                    "peak_memory_mb": e.peak_memory_mb,
                    "load_time_s": e.load_time_s,
                    "error": e.error,
                }
                for e in run.deployability
            ],
            "models": {
                name: {
                    "kind": r.kind,
                    "accuracy": r.accuracy,
                    "median_latency_ms": r.median_latency_ms if r.latencies_ms else None,
                    "p10_latency_ms": r.p10_latency_ms if r.latencies_ms else None,
                    "p90_latency_ms": r.p90_latency_ms if r.latencies_ms else None,
                    "deployable": r.deployable,
                    "deploy_error": r.deploy_error,
                }
                for name, r in run.model_results.items()
            },
            "ecr": {
                "value": run.ecr_result.value if math.isfinite(run.ecr_result.value) else str(run.ecr_result.value),
                "delta_acc": run.ecr_result.delta_acc,
                "lat_mmllm_ms": run.ecr_result.lat_mmllm_ms,
                "lat_classic_ms": run.ecr_result.lat_classic_ms,
                "best_mmllm": run.ecr_result.best_mmllm,
                "best_classic": run.ecr_result.best_classic,
            } if run.ecr_result else None,
        }
        data.append(entry)
    Path(path).write_text(json.dumps(data, indent=2))
    print(f"\nResults saved → {path}")
