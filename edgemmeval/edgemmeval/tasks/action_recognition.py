from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any, List

import torch
import ollama

import cv2

from PIL import Image
import io

from ..task import ModelConfig, Task

_N_FRAMES = 8
_FRAME_SIZE = (112, 112)


class ActionRecognitionTask(Task):
    name = "action_recognition"

    def __init__(self, data_dir: str, max_samples: int = 50):
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples

    def load_dataset(self) -> List[tuple]:
        samples = []
        for cls_dir in sorted(self.data_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = cls_dir.name.replace("_", " ")
            for video in sorted(cls_dir.glob("*.mp4")):
                samples.append((str(video), label))
                if len(samples) >= self.max_samples:
                    return samples
        return samples

    def score(self, prediction: str, ground_truth: str) -> float:
        return 1.0 if prediction.lower().strip() == ground_truth.lower().strip() else 0.0

    def classic_models(self) -> List[ModelConfig]:
        all_labels = [
            d.name.replace("_", " ")
            for d in self.data_dir.iterdir()
            if d.is_dir()
        ]
        return [
            ModelConfig(
                name="r3d-18",
                kind="classic",
                loader=lambda: _load_video_model("r3d_18", all_labels),
                infer=_infer_video_model,
            ),
            ModelConfig(
                name="mc3-18",
                kind="classic",
                loader=lambda: _load_video_model("mc3_18", all_labels),
                infer=_infer_video_model,
            ),
            ModelConfig(
                name="r2plus1d-18",
                kind="classic",
                loader=lambda: _load_video_model("r2plus1d_18", all_labels),
                infer=_infer_video_model,
            ),
        ]

    def mmllm_models(self) -> List[ModelConfig]:
        all_labels = [
            d.name.replace("_", " ")
            for d in self.data_dir.iterdir()
            if d.is_dir()
        ]
        label_str = ", ".join(all_labels[:50])
        prompt = (
            f"These frames are from a video clip. "
            f"Identify the action being performed. "
            f"Choose from: {label_str}. "
            "Reply with only the action label, nothing else."
        )
        return [
            ModelConfig(
                name="gemma3n-e2b",
                kind="mmllm",
                loader=lambda: _ollama_handle("gemma3n:e2b"),
                infer=lambda model, path: _infer_ollama_video(model, path, prompt, all_labels),
            ),
            ModelConfig(
                name="gemma3n-e4b",
                kind="mmllm",
                loader=lambda: _ollama_handle("gemma3n:e4b"),
                infer=lambda model, path: _infer_ollama_video(model, path, prompt, all_labels),
            ),
        ]


def _load_video_model(model_name: str, label_list: List[str]):
    from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
    from torchvision.models.video import R3D_18_Weights, MC3_18_Weights, R2Plus1D_18_Weights
    import torch

    model_map = {
        "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
        "mc3_18": (mc3_18, MC3_18_Weights.DEFAULT),
        "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights.DEFAULT),
    }
    fn, weights = model_map[model_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = fn(weights=weights).to(device).eval()
    kinetics_labels = weights.meta["categories"]
    return (m, kinetics_labels, label_list, device)


def _infer_video_model(model_tuple, video_path: str) -> str:
    import torchvision.transforms as T
    m, kinetics_labels, label_list, device = model_tuple
    frames = _sample_frames_tensor(video_path, n_frames=_N_FRAMES, size=_FRAME_SIZE)
    frames = frames.to(device)
    with torch.no_grad():
        out = m(frames)
    idx = out.argmax(1).item()
    kinetics_pred = kinetics_labels[idx].lower()
    # Try to match to our dataset labels
    for label in label_list:
        if label.lower() in kinetics_pred or kinetics_pred in label.lower():
            return label
    return kinetics_pred


def _sample_frames_tensor(video_path: str, n_frames: int, size: tuple):
    """
    Returns a (1, C, T, H, W) float32 tensor normalised for torchvision video models.
    Falls back to OpenCV if torchvision.io.read_video is unavailable.
    """
    try:
        import torchvision.io as tvio
        import torchvision.transforms.functional as TF
        vframes, _, _ = tvio.read_video(video_path, pts_unit="sec")
        indices = _uniform_indices(len(vframes), n_frames)
        frames = vframes[indices]
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
    except Exception:
        frames = _read_with_cv2(video_path, n_frames)

    # Resize
    import torch.nn.functional as F
    frames = F.interpolate(frames, size=size)
    # Normalise (Kinetics mean/std)
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    return frames.unsqueeze(0).permute(0, 2, 1, 3, 4)


def _read_with_cv2(video_path: str, n_frames: int):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(_uniform_indices(total, n_frames))
    frames = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
    cap.release()
    return torch.stack(frames)


def _uniform_indices(total: int, n: int) -> List[int]:
    if total <= n:
        return list(range(total))
    step = total / n
    return [int(i * step) for i in range(n)]

def _ollama_handle(model_tag: str) -> str:
    available = {m["name"] for m in ollama.list()["models"]}
    if not any(model_tag in n for n in available):
        raise RuntimeError(f"Ollama model '{model_tag}' not found. Run: ollama pull {model_tag}")
    return model_tag


def _infer_ollama_video(model_tag: str, video_path: str, prompt: str, labels: List[str]) -> str:
    """Sample frames from the video and send as images to the MMLLM."""
    frames = _sample_frames_tensor(video_path, n_frames=4, size=(224, 224))
    # Convert back to PIL for base64 encoding
    t = frames.squeeze(0).permute(1, 2, 3, 0)  # (T, H, W, C)
    images_b64 = []
    for i in range(t.shape[0]):
        arr = (t[i].numpy() * 255).clip(0, 255).astype("uint8")
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        images_b64.append(base64.b64encode(buf.getvalue()).decode())

    resp = ollama.chat(
        model=model_tag,
        messages=[{"role": "user", "content": prompt, "images": images_b64}],
        options={"temperature": 0},
    )
    raw = resp["message"]["content"].strip().lower()
    for label in labels:
        if label.lower() in raw:
            return label
    return raw
