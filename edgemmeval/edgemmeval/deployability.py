"""
Deployability probing: can a model load on the current device?
Records peak memory delta and load time.
"""
from __future__ import annotations

import gc
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


@dataclass
class DeployabilityEntry:
    model_name: str
    kind: str
    hardware_id: str
    deployable: bool
    peak_memory_mb: Optional[float] = None
    load_time_s: Optional[float] = None
    error: str = ""

    def status_str(self) -> str:
        if not self.deployable:
            return f"FAIL  ({self.error[:60]})"
        mem = f"  peak={self.peak_memory_mb:.0f}MB" if self.peak_memory_mb else ""
        t = f"  load={self.load_time_s:.1f}s" if self.load_time_s else ""
        return f"OK{mem}{t}"


def probe(model_cfg, hardware_id: str) -> DeployabilityEntry:
    """Attempt to load a model; record memory, timing, and any failure."""
    gc.collect()
    mem_before = _rss_mb()
    t0 = time.perf_counter()
    try:
        model = model_cfg.loader()
        load_time = time.perf_counter() - t0
        peak_delta = max(0.0, _rss_mb() - mem_before)
        del model
        gc.collect()
        return DeployabilityEntry(
            model_name=model_cfg.name,
            kind=model_cfg.kind,
            hardware_id=hardware_id,
            deployable=True,
            peak_memory_mb=peak_delta,
            load_time_s=load_time,
        )
    except Exception as exc:
        return DeployabilityEntry(
            model_name=model_cfg.name,
            kind=model_cfg.kind,
            hardware_id=hardware_id,
            deployable=False,
            error=str(exc),
        )


def current_hardware_id() -> str:
    """Best-effort hardware identifier for the running machine."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        gpu = out.decode().strip().splitlines()[0].strip()
        if gpu:
            return gpu
    except Exception:
        pass
    return platform.node() or "unknown"


def _rss_mb() -> float:
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    return 0.0
