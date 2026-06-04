from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..task import ModelConfig, Task


class ObjectCountingTask(Task):
    name = "object_counting"

    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        target_class: str = "person",
        max_samples: int = 500,
    ):
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self.target_class = target_class
        self.max_samples = max_samples
        self._coco_id: Optional[int] = None

    def load_dataset(self) -> List[tuple]:
        import json
        ann = json.loads(self.annotations_file.read_text())

        # Find category id for target class
        cat_id = next(
            (c["id"] for c in ann["categories"] if c["name"] == self.target_class),
            None,
        )
        if cat_id is None:
            raise ValueError(f"Class '{self.target_class}' not found in COCO categories")
        self._coco_id = cat_id

        counts: Dict[int, int] = {}
        for a in ann["annotations"]:
            if a["category_id"] == cat_id:
                counts[a["image_id"]] = counts.get(a["image_id"], 0) + 1

        samples = []
        for img in ann["images"]:
            iid = img["id"]
            if iid not in counts:
                continue
            img_path = self.images_dir / img["file_name"]
            if img_path.exists():
                samples.append((str(img_path), counts[iid]))
            if len(samples) >= self.max_samples:
                break
        return samples

    def score(self, prediction: int, ground_truth: int) -> float:
        return 1.0 if int(prediction) == int(ground_truth) else 0.0

    def classic_models(self) -> List[ModelConfig]:
        cls = self.target_class
        return [
            ModelConfig(
                name="yolov8",
                kind="classic",
                loader=lambda: _load_yolo("yolov8n.pt"),
                infer=lambda model, path: _count_yolo(model, path, cls),
            ),
            ModelConfig(
                name="faster-rcnn-r50",
                kind="classic",
                loader=_load_fasterrcnn,
                infer=lambda model, path: _count_torchvision_det(model, path, cls),
            ),
            ModelConfig(
                name="detr-r50",
                kind="classic",
                loader=_load_detr,
                infer=lambda model, path: _count_torchvision_det(model, path, cls),
            ),
        ]

    def mmllm_models(self) -> List[ModelConfig]:
        cls = self.target_class
        prompt = (
            f"Count the number of {cls} objects visible in this image. "
            "Reply with only the integer number, nothing else."
        )
        return [
            ModelConfig(
                name="blip-2",
                kind="mmllm",
                loader=_load_blip2,
                infer=lambda model, path: _count_blip2(model, path, cls),
            ),
            ModelConfig(
                name="gemma3n-e2b",
                kind="mmllm",
                loader=lambda: _ollama_handle("gemma3n:e2b"),
                infer=lambda model, path: _count_ollama(model, path, prompt),
            ),
            ModelConfig(
                name="gemma3n-e4b",
                kind="mmllm",
                loader=lambda: _ollama_handle("gemma3n:e4b"),
                infer=lambda model, path: _count_ollama(model, path, prompt),
            ),
            ModelConfig(
                name="qwen2-vl-2b",
                kind="mmllm",
                loader=lambda: _ollama_handle("qwen2.5vl:3b"),
                infer=lambda model, path: _count_ollama(model, path, prompt),
            ),
        ]

_COCO91_NAMES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    16: "bird", 17: "cat", 18: "dog",
}
_NAME_TO_COCO91 = {v: k for k, v in _COCO91_NAMES.items()}

def _load_yolo(weights: str):
    from ultralytics import YOLO
    return YOLO(weights)


def _count_yolo(model, img_path: str, target_class: str) -> int:
    results = model(img_path, verbose=False)
    count = 0
    for r in results:
        for cls_id in r.boxes.cls.cpu().numpy().astype(int):
            if model.names[cls_id].lower() == target_class.lower():
                count += 1
    return count

def _load_fasterrcnn():
    import torch
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights,
    )
    w = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = fasterrcnn_resnet50_fpn_v2(weights=w).to(device).eval()
    return (m, device)


def _load_detr():
    import torch
    from torchvision.models.detection import (
        detr_resnet50,
        DETR_ResNet50_Weights,
    )
    w = DETR_ResNet50_Weights.DEFAULT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = detr_resnet50(weights=w).to(device).eval()
    return (m, device)


def _count_torchvision_det(model_tuple, img_path: str, target_class: str, score_thresh: float = 0.5) -> int:
    import torch
    from PIL import Image
    import torchvision.transforms.functional as TF
    model, device = model_tuple
    img = Image.open(img_path).convert("RGB")
    tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(tensor)[0]
    target_id = _NAME_TO_COCO91.get(target_class.lower())
    count = 0
    for label, score in zip(preds["labels"].cpu().numpy(), preds["scores"].cpu().numpy()):
        if score >= score_thresh and (target_id is None or int(label) == target_id):
            count += 1
    return count

def _load_blip2():
    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    return (model, proc, device)


def _count_blip2(model_tuple, img_path: str, target_class: str) -> int:
    import torch
    from PIL import Image
    model, proc, device = model_tuple
    img = Image.open(img_path).convert("RGB")
    prompt = f"How many {target_class} are in this image? Answer with a number only."
    inputs = proc(img, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10)
    text = proc.decode(out[0], skip_special_tokens=True)
    return _parse_int(text)

def _ollama_handle(model_tag: str) -> str:
    import ollama
    available = {m["name"] for m in ollama.list()["models"]}
    if not any(model_tag in n for n in available):
        raise RuntimeError(f"Ollama model '{model_tag}' not found. Run: ollama pull {model_tag}")
    return model_tag


def _count_ollama(model_tag: str, img_path: str, prompt: str) -> int:
    import ollama
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = ollama.chat(
        model=model_tag,
        messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
        options={"temperature": 0},
    )
    return _parse_int(resp["message"]["content"])


def _parse_int(text: str) -> int:
    m = re.search(r"\d+", str(text))
    return int(m.group()) if m else 0
