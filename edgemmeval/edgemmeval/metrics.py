"""
Metric utilities: ECR, WER, accuracy helpers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

def ecr(delta_acc: float, lat_mmllm_ms: float, lat_classic_ms: float) -> float:
    """
    ECR = ΔAcc / log10(L_MMLLM / L_classic)

    Positive ECR: MMLLM improves accuracy per order-of-magnitude latency cost.
    Negative ECR: classic model dominates on both axes.
    ECR = +inf: MMLLM is faster AND more accurate (shouldn't normally occur).
    """
    if lat_classic_ms <= 0 or lat_mmllm_ms <= 0:
        raise ValueError("Latencies must be positive")
    ratio = lat_mmllm_ms / lat_classic_ms
    if ratio <= 1.0:
        return math.inf if delta_acc > 0 else -math.inf
    return delta_acc / math.log10(ratio)


@dataclass
class ECRResult:
    task_name: str
    best_mmllm: str
    best_classic: str
    delta_acc: float
    lat_mmllm_ms: float
    lat_classic_ms: float
    value: float  # the ECR score

    def __str__(self) -> str:
        sign = "+" if self.value >= 0 else ""
        return (
            f"{self.task_name:<22} "
            f"ΔAcc={self.delta_acc:+.3f}  "
            f"L_MMLLM={self.lat_mmllm_ms:>8.1f}ms  "
            f"L_classic={self.lat_classic_ms:>7.1f}ms  "
            f"ECR={sign}{self.value:.3f}"
        )


def accuracy(scores: List[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein-based WER at word level."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            prev, d[j] = d[j], min(
                d[j] + 1,
                d[j - 1] + 1,
                prev + (0 if ref[i - 1] == hyp[j - 1] else 1),
            )
    return d[m] / n


def exact_match(prediction: Any, ground_truth: Any) -> float:
    return 1.0 if str(prediction).strip().lower() == str(ground_truth).strip().lower() else 0.0
