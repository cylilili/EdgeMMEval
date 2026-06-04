from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..task import ModelConfig, Task

_OCR_PROMPT = (
    "You are a clinical document parser. Extract ONLY the medication names from "
    "this prescription image. Return a JSON object with a single key 'medications' "
    "whose value is a list of objects each with a 'name' field containing the "
    "full canonical drug name in lowercase. Example: "
    '{{"medications":[{{"name":"amoxicillin"}},{{"name":"ibuprofen"}}]}}. '
    "Output only valid JSON, nothing else."
)


class OCRTask(Task):
    """Prescription drug-name extraction (OCR extensibility demo)."""

    name = "ocr"

    def __init__(self, images_dir: str, labels_file: str, max_samples: int = 200):
        """
        images_dir   : directory containing prescription image files (PNG/JPG)
        labels_file  : JSON file with ground-truth drug names, keyed by image id
                       Expected format: [{"id": "...", "gold": {"canonical_names": [...]}}]
        max_samples  : cap evaluation to first N samples
        """
        self.images_dir = Path(images_dir)
        self.labels_file = Path(labels_file)
        self.max_samples = max_samples

    def load_dataset(self) -> List[Tuple[str, Any]]:
        with open(self.labels_file) as f:
            records = json.load(f)
        samples = []
        for rec in records[: self.max_samples]:
            
            parts = rec["id"].rsplit("_", 1)
            scenario = parts[-1] if len(parts) > 1 else "baseline"
            stem = "_".join(rec["id"].split("_")[:2])
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = self.images_dir / scenario / (stem + ext)
                if candidate.exists():
                    break
            else:
                candidate = self.images_dir / (rec["id"] + ".png")
            gold = set(rec["gold"]["canonical_names"])
            samples.append((str(candidate), gold))
        return samples

    def score(self, prediction: Any, ground_truth: Any) -> float:
        """Full accuracy: 1.0 iff all gold names are present in prediction."""
        if isinstance(prediction, set):
            pred_names = prediction
        elif isinstance(prediction, list):
            pred_names = {n.lower().strip() for n in prediction}
        else:
            pred_names = set()
        gold = {n.lower().strip() for n in ground_truth}
        return 1.0 if gold.issubset(pred_names) else 0.0

    def classic_models(self) -> List[ModelConfig]:
        return [
            ModelConfig(
                name="tesseract",
                kind="classic",
                loader=_load_tesseract,
                infer=_infer_tesseract,
            ),
            ModelConfig(
                name="easyocr",
                kind="classic",
                loader=_load_easyocr,
                infer=_infer_easyocr,
            ),
        ]

    def mmllm_models(self) -> List[ModelConfig]:
        tags = [
            "deepseek-ocr:3b",
            "llava:7b",
            "llama3.2-vision:11b",
            "gemma3:4b",
            "qwen3-vl:2b",
        ]
        configs = []
        for tag in tags:
            t = tag
            configs.append(ModelConfig(
                name=tag,
                kind="mmllm",
                loader=lambda tag=t: _ollama_handle(tag),
                infer=lambda model, path, tag=t: _infer_ollama_image(model, path),
            ))
        return configs

def _load_tesseract():
    import pytesseract
    return "tesseract"


def _infer_tesseract(model, image_path: str) -> set:
    import re
    import pytesseract
    from PIL import Image
    text = pytesseract.image_to_string(Image.open(image_path))
    return _extract_drug_names_heuristic(text)

def _load_easyocr():
    import easyocr
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def _infer_easyocr(reader, image_path: str) -> set:
    results = reader.readtext(image_path, detail=0)
    return _extract_drug_names_heuristic(" ".join(results))

_COMMON_DRUG_WORDS = frozenset(
    [
        "mg", "tab", "tabs", "tablet", "tablets", "cap", "caps", "capsule",
        "dose", "daily", "once", "twice", "three", "oral", "po", "iv", "pr",
        "qd", "bid", "tid", "qid", "qam", "qpm", "prn", "hs", "days",
        "weeks", "month", "rx", "dr", "date", "sig", "refill", "no",
        "dispense", "quantity",
    ]
)


def _extract_drug_names_heuristic(text: str) -> set:
    import re
    words = re.findall(r"[a-zA-Z]{4,}", text)
    candidates = set()
    for w in words:
        lw = w.lower()
        if lw not in _COMMON_DRUG_WORDS:
            candidates.add(lw)
    return candidates

def _ollama_handle(model_tag: str) -> str:
    import ollama
    available = {m["name"] for m in ollama.list()["models"]}
    if not any(model_tag in n for n in available):
        raise RuntimeError(
            f"Ollama model '{model_tag}' not found. Run: ollama pull {model_tag}"
        )
    return model_tag


def _infer_ollama_image(model_tag: str, image_path: str) -> set:
    import json
    import re
    import ollama
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = ollama.chat(
        model=model_tag,
        messages=[{"role": "user", "content": _OCR_PROMPT, "images": [img_b64]}],
        options={"temperature": 0},
    )
    raw = resp["message"]["content"].strip()
    names: set = set()
    try:
        # Try to parse JSON directly
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            meds = data.get("medications", [])
            for m in meds:
                n = m.get("name", "").lower().strip()
                if n:
                    names.add(n)
    except (json.JSONDecodeError, AttributeError):
        # Fall back to word extraction
        names = _extract_drug_names_heuristic(raw)
    return names
