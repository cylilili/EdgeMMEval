"""
Abstract Task interface for the edge MMLLM benchmark harness.

To add a new task, subclass Task and implement:
  - name: str identifier
  - load_dataset() -> list of (input, ground_truth)
  - score(prediction, ground_truth) -> float in [0, 1]
  - classic_models() -> list[ModelConfig]
  - mmllm_models()   -> list[ModelConfig]
"""
from __future__ import annotations

import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class ModelConfig:
    """Everything the harness needs to load and run one model."""
    name: str
    kind: str                        
    loader: Callable[[], Any]        
    infer: Callable[[Any, Any], Any]


@dataclass
class ModelResult:
    model_name: str
    kind: str
    accuracy: float
    latencies_ms: List[float]
    deployable: bool = True
    deploy_error: str = ""

    @property
    def median_latency_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else float("inf")

    @property
    def p10_latency_ms(self) -> float:
        if not self.latencies_ms:
            return float("inf")
        s = sorted(self.latencies_ms)
        return s[max(0, int(0.10 * len(s)) - 1)]

    @property
    def p90_latency_ms(self) -> float:
        if not self.latencies_ms:
            return float("inf")
        s = sorted(self.latencies_ms)
        return s[int(0.90 * len(s))]


class Task(ABC):
    name: str = "unnamed"

    @abstractmethod
    def load_dataset(self) -> List[tuple]:
        """Return list of (input, ground_truth) pairs."""
        ...

    @abstractmethod
    def score(self, prediction: Any, ground_truth: Any) -> float:
        """Per-sample score in [0, 1]. Higher is better."""
        ...

    @abstractmethod
    def classic_models(self) -> List[ModelConfig]:
        ...

    @abstractmethod
    def mmllm_models(self) -> List[ModelConfig]:
        ...
