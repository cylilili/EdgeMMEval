"""
Memory envelope estimation for MMLLMs and classic models.

Provides per-model memory footprint estimates and device compatibility
checks, enabling practitioners to determine which hardware tiers can
serve a given model before running any inference.

NOTE: All memory figures are principled estimates based on known model
architectures (fp16_gb ≈ params_b * 2, with vision encoder overhead added
for multimodal models). They are NOT measured values.
"""
from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Model catalog
# Each entry: params_b  – parameter count in billions (estimate)
#             fp16_gb   – weight footprint at fp16 (estimate)
#             kvcache_gb – additional KV-cache / activation overhead (estimate)
#             note      – brief rationale / architecture hint
# ---------------------------------------------------------------------------
MODEL_CATALOG: dict[str, dict] = {
    # --- Classic / non-LLM models ---
    "tesseract": {
        "params_b": 0.0,
        "fp16_gb": 0.10,
        "kvcache_gb": 0.0,
        "note": "CPU-only OCR; no GPU needed",
    },
    "easyocr": {
        "params_b": 0.01,
        "fp16_gb": 0.20,
        "kvcache_gb": 0.0,
        "note": "ResNet+LSTM; ~200MB",
    },
    "mobilenet_v3": {
        "params_b": 0.005,
        "fp16_gb": 0.01,
        "kvcache_gb": 0.0,
        "note": "MobileNetV3-Large",
    },
    "resnet50": {
        "params_b": 0.025,
        "fp16_gb": 0.05,
        "kvcache_gb": 0.0,
        "note": "ResNet-50",
    },
    "convnext_tiny": {
        "params_b": 0.028,
        "fp16_gb": 0.06,
        "kvcache_gb": 0.0,
        "note": "ConvNeXt-Tiny",
    },
    "yolov8n": {
        "params_b": 0.003,
        "fp16_gb": 0.01,
        "kvcache_gb": 0.0,
        "note": "YOLOv8 nano",
    },
    # --- Text-only MMLLMs ---
    "qwen3.5:0.8b": {
        "params_b": 0.8,
        "fp16_gb": 1.6,
        "kvcache_gb": 0.4,
        "note": "text-only; smallest Qwen3.5",
    },
    "qwen3.5:2b": {
        "params_b": 2.0,
        "fp16_gb": 4.1,
        "kvcache_gb": 1.0,
        "note": "",
    },
    "deepseek-ocr:3b": {
        "params_b": 3.0,
        "fp16_gb": 6.2,
        "kvcache_gb": 1.5,
        "note": "OCR-specialized",
    },
    # --- Vision-language MMLLMs ---
    "qwen3-vl:2b": {
        "params_b": 2.0,
        "fp16_gb": 4.5,
        "kvcache_gb": 2.0,
        "note": "+vision encoder ~2GB",
    },
    "gemma3n:e2b": {
        "params_b": 2.0,
        "fp16_gb": 4.5,
        "kvcache_gb": 1.5,
        "note": "Gemma3n nano E2B",
    },
    "gemma3n:e4b": {
        "params_b": 4.0,
        "fp16_gb": 8.5,
        "kvcache_gb": 2.5,
        "note": "Gemma3n nano E4B",
    },
    "gemma3:4b": {
        "params_b": 4.0,
        "fp16_gb": 8.2,
        "kvcache_gb": 2.0,
        "note": "",
    },
    "medgemma-1.5-4b": {
        "params_b": 4.0,
        "fp16_gb": 8.5,
        "kvcache_gb": 2.0,
        "note": "medical-tuned Gemma",
    },
    "llava:7b": {
        "params_b": 7.0,
        "fp16_gb": 14.5,
        "kvcache_gb": 3.5,
        "note": "+CLIP vision encoder",
    },
    "llama3.2-vision:11b": {
        "params_b": 11.0,
        "fp16_gb": 23.0,
        "kvcache_gb": 5.5,
        "note": "",
    },
}

DEVICE_CATALOG: dict[str, dict] = {
    "Jetson Orin": {
        "total_gb": 8,
        "memory_type": "unified",
    },
    "Jetson Orin NX": {
        "total_gb": 16,
        "memory_type": "unified",
    },
    "RTX PRO 6000": {
        "total_gb": 96,
        "memory_type": "dedicated_gpu",
    },
    "DGX Spark": {
        "total_gb": 128,
        "memory_type": "unified",
    },
}

def required_memory_gb(model_name: str, headroom: float = 0.15) -> float:
    """
    Return the total inference memory footprint for *model_name*.

    The estimate is fp16_gb + kvcache_gb; headroom is accepted for API
    consistency but is not applied here (apply it at the device level).
    Returns 0.1 if the model is not in MODEL_CATALOG.
    """
    entry = MODEL_CATALOG.get(model_name)
    if entry is None:
        return 0.1
    return entry["fp16_gb"] + entry["kvcache_gb"]


def fits_on_device(
    model_name: str,
    device_name: str,
    headroom: float = 0.15,
) -> str:
    """
    Check whether *model_name* fits on *device_name*.

    Returns one of:
        "ok"    – fits comfortably (< 80 % of available memory)
        "tight" – fits but uses 80–100 % of available memory
        "oom"   – does not fit
    Available memory = total_gb * (1 - headroom).
    """
    device = DEVICE_CATALOG.get(device_name)
    if device is None:
        raise ValueError(f"Unknown device: {device_name!r}")

    available_gb = device["total_gb"] * (1.0 - headroom)
    needed_gb = required_memory_gb(model_name, headroom=headroom)

    if needed_gb > available_gb:
        return "oom"
    if needed_gb > 0.80 * available_gb:
        return "tight"
    return "ok"


def memory_envelope_df() -> pd.DataFrame:
    """
    Return a DataFrame summarising the memory envelope for all catalogued models.

    Columns: model, params_b, fp16_gb, total_inference_gb, <device> …
    The device columns contain "ok", "tight", or "oom".
    """
    rows = []
    for model_name, entry in MODEL_CATALOG.items():
        row: dict = {
            "model": model_name,
            "params_b": entry["params_b"],
            "fp16_gb": entry["fp16_gb"],
            "total_inference_gb": entry["fp16_gb"] + entry["kvcache_gb"],
        }
        for device_name in DEVICE_CATALOG:
            row[device_name] = fits_on_device(model_name, device_name)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("total_inference_gb").reset_index(drop=True)
    return df
