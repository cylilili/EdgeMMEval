from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, List

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models

from ..task import ModelConfig, Task

_WN_TO_LABEL = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}
_LABELS = list(_WN_TO_LABEL.values())

_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageClassificationTask(Task):
    name = "image_classification"

    def __init__(self, data_dir: str, split: str = "val", max_per_class: int = 50):
        self.data_dir = Path(data_dir) / split
        self.max_per_class = max_per_class

    def load_dataset(self) -> List[tuple]:
        samples = []
        for cls_dir in sorted(self.data_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = _WN_TO_LABEL.get(cls_dir.name, cls_dir.name)
            imgs = sorted(cls_dir.glob("*.JPEG")) + sorted(cls_dir.glob("*.jpg"))
            for p in imgs[: self.max_per_class]:
                samples.append((str(p), label))
        return samples

    def score(self, prediction: str, ground_truth: str) -> float:
        return 1.0 if prediction.lower().strip() == ground_truth.lower().strip() else 0.0

    def classic_models(self) -> List[ModelConfig]:
        return [
            ModelConfig(
                name="mobilenet_v3_large",
                kind="classic",
                loader=lambda: _load_torchvision(
                    models.mobilenet_v3_large,
                    models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
                ),
                infer=_infer_torchvision,
            ),
            ModelConfig(
                name="resnet50",
                kind="classic",
                loader=lambda: _load_torchvision(
                    models.resnet50,
                    models.ResNet50_Weights.IMAGENET1K_V2,
                ),
                infer=_infer_torchvision,
            ),
            ModelConfig(
                name="convnext_tiny",
                kind="classic",
                loader=lambda: _load_torchvision(
                    models.convnext_tiny,
                    models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                ),
                infer=_infer_torchvision,
            ),
        ]

    def mmllm_models(self) -> List[ModelConfig]:
        label_str = ", ".join(_LABELS)
        prompt = (
            f"Classify this image into exactly one of: {label_str}. "
            "Reply with only the category name, nothing else."
        )
        return [
            ModelConfig(
                name="gemma3n-e2b",
                kind="mmllm",
                loader=lambda: _ollama_handle("gemma3n:e2b"),
                infer=lambda model, path: _infer_ollama(model, path, prompt, _LABELS),
            ),
            ModelConfig(
                name="gemma3n-e4b",
                kind="mmllm",
                loader=lambda: _ollama_handle("gemma3n:e4b"),
                infer=lambda model, path: _infer_ollama(model, path, prompt, _LABELS),
            ),
            ModelConfig(
                name="qwen3-vl-2b",
                kind="mmllm",
                loader=lambda: _ollama_handle("qwen2.5vl:3b"),
                infer=lambda model, path: _infer_ollama(model, path, prompt, _LABELS),
            ),
            ModelConfig(
                name="ministral-3b",
                kind="mmllm",
                loader=lambda: _ollama_handle("mistral:3b-instruct"),
                infer=lambda model, path: _infer_ollama_text(model, path, prompt, _LABELS),
            ),
        ]

def _load_torchvision(model_fn, weights):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = model_fn(weights=weights).to(device).eval()
    cats = weights.meta["categories"]
    return (m, cats, device)


def _infer_torchvision(model_tuple, img_path: str) -> str:
    model, categories, device = model_tuple
    img = Image.open(img_path).convert("RGB")
    tensor = _TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    imagenet_label = categories[out.argmax(1).item()].lower()
    for label in _LABELS:
        if label.lower() in imagenet_label or imagenet_label in label.lower():
            return label
    return imagenet_label


def _ollama_handle(model_tag: str) -> str:
    """Validate model is available in Ollama; return tag as handle."""
    import ollama
    available = {m["name"] for m in ollama.list()["models"]}
    if not any(model_tag in n for n in available):
        raise RuntimeError(
            f"Ollama model '{model_tag}' not found. Run: ollama pull {model_tag}"
        )
    return model_tag


def _infer_ollama(model_tag: str, img_path: str, prompt: str, labels: List[str]) -> str:
    import ollama
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = ollama.chat(
        model=model_tag,
        messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
        options={"temperature": 0},
    )
    raw = resp["message"]["content"].strip().lower()
    # Match to closest label
    for label in labels:
        if label.lower() in raw:
            return label
    return raw


def _infer_ollama_text(model_tag: str, img_path: str, prompt: str, labels: List[str]) -> str:
    """Text-only fallback for models that do not support vision."""
    import ollama
    resp = ollama.chat(
        model=model_tag,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    raw = resp["message"]["content"].strip().lower()
    for label in labels:
        if label.lower() in raw:
            return label
    return raw
