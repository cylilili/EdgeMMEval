from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, List

from ..task import ModelConfig, Task

_FRAMES_PER_VIDEO = 20


class HazardDetectionTask(Task):
    name = "hazard_detection"

    def __init__(self, data_dir: str, max_videos: int = 200, k_periodic: int = 5):
        """
        data_dir      : DetectiumFire root; expects fire_videos/ and no_fire_videos/
        max_videos    : cap total videos evaluated (split evenly between classes)
        k_periodic    : invocation frequency for the periodic hybrid strategy
        """
        self.data_dir = Path(data_dir)
        self.max_videos = max_videos
        self.k_periodic = k_periodic

    def load_dataset(self) -> List[tuple]:
        """Returns (video_path, ground_truth_label) where label ∈ {"fire", "no_fire"}."""
        samples = []
        per_class = self.max_videos // 2
        for label, subdir in [("fire", "fire_videos"), ("no_fire", "no_fire_videos")]:
            d = self.data_dir / subdir
            if not d.exists():
                continue
            for video in sorted(d.glob("*.mp4"))[:per_class]:
                samples.append((str(video), label))
        return samples

    def score(self, prediction: str, ground_truth: str) -> float:
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0

    def classic_models(self) -> List[ModelConfig]:
        return [
            ModelConfig(
                name="yolov8-fire",
                kind="classic",
                loader=_load_yolo_fire,
                infer=_infer_yolo_video,
            ),
        ]

    def mmllm_models(self) -> List[ModelConfig]:
        k = self.k_periodic
        return [
            ModelConfig(
                name="gemma3n-e2b",
                kind="mmllm",
                loader=lambda: _ollama_handle("gemma3n:e2b"),
                infer=lambda model, path: _infer_ollama_video(model, path),
            ),
            ModelConfig(
                name=f"periodic-k{k}-gemma3n-e2b",
                kind="mmllm",
                loader=lambda: _load_periodic_hybrid(k),
                infer=lambda model, path: _infer_periodic_hybrid(model, path),
            ),
        ]


def _load_yolo_fire():
    from ultralytics import YOLO
    candidates = [
        "yolov8-fire.pt",
        Path.home() / ".cache" / "yolo" / "yolov8-fire.pt",
    ]
    for c in candidates:
        if Path(c).exists():
            return YOLO(str(c))
        
    return YOLO("yolov8n.pt")


def _infer_yolo_video(model, video_path: str) -> str:
    """Sample frames; predict fire/no_fire by majority vote."""
    frames = _sample_frame_paths(video_path, _FRAMES_PER_VIDEO)
    fire_votes = 0
    for frame_path in frames:
        results = model(frame_path, verbose=False)
        for r in results:
            names = [model.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
            if any("fire" in n.lower() or "flame" in n.lower() for n in names):
                fire_votes += 1
    return "fire" if fire_votes > len(frames) / 2 else "no_fire"


_FIRE_PROMPT = (
    "Is there real fire or active flames visible in this image? "
    "Do NOT count orange lighting, sunsets, fire on screens, or reflections. "
    "Reply with exactly 'fire' or 'no_fire', nothing else."
)


def _ollama_handle(model_tag: str) -> str:
    import ollama
    available = {m["name"] for m in ollama.list()["models"]}
    if not any(model_tag in n for n in available):
        raise RuntimeError(f"Ollama model '{model_tag}' not found. Run: ollama pull {model_tag}")
    return model_tag


def _infer_ollama_video(model_tag: str, video_path: str) -> str:
    """Run MMLLM on sampled frames; majority vote."""
    import ollama
    frames = _sample_frame_paths(video_path, _FRAMES_PER_VIDEO)
    fire_votes = 0
    for frame_path in frames:
        with open(frame_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        resp = ollama.chat(
            model=model_tag,
            messages=[{"role": "user", "content": _FIRE_PROMPT, "images": [img_b64]}],
            options={"temperature": 0},
        )
        raw = resp["message"]["content"].strip().lower()
        if "no_fire" in raw or "no fire" in raw:
            pass
        elif "fire" in raw:
            fire_votes += 1
    return "fire" if fire_votes > len(frames) / 2 else "no_fire"


def _load_periodic_hybrid(k: int):
    from ultralytics import YOLO
    yolo = _load_yolo_fire()
    mmllm_tag = _ollama_handle("gemma3n:e2b")
    return (yolo, mmllm_tag, k)


def _infer_periodic_hybrid(model_tuple, video_path: str) -> str:
    """
    YOLOv8 on every frame; gemma3n every k-th frame.
    MMLLM decision overrides YOLO on frames it is called for.
    Majority vote over all frames.
    """
    import ollama
    yolo, mmllm_tag, k = model_tuple
    frames = _sample_frame_paths(video_path, _FRAMES_PER_VIDEO)
    fire_votes = 0
    for i, frame_path in enumerate(frames):
        if i % k == 0:
            # MMLLM call
            with open(frame_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            resp = ollama.chat(
                model=mmllm_tag,
                messages=[{"role": "user", "content": _FIRE_PROMPT, "images": [img_b64]}],
                options={"temperature": 0},
            )
            raw = resp["message"]["content"].strip().lower()
            if "no_fire" not in raw and "no fire" not in raw and "fire" in raw:
                fire_votes += 1
        else:
            # YOLO call
            results = yolo(frame_path, verbose=False)
            for r in results:
                names = [yolo.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
                if any("fire" in n.lower() or "flame" in n.lower() for n in names):
                    fire_votes += 1
                    break
    return "fire" if fire_votes > len(frames) / 2 else "no_fire"

def _sample_frame_paths(video_path: str, n_frames: int) -> List[str]:
    """Extract n_frames uniformly from video; save as temp JPEGs; return paths."""
    import cv2
    import tempfile
    import os
    cap = cv2.VideoCapture(video_path)
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = set(_uniform_indices(total, n_frames))
    paths = []
    tmpdir = tempfile.mkdtemp(prefix="hd_frames_")
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            out_path = os.path.join(tmpdir, f"frame_{i:05d}.jpg")
            cv2.imwrite(out_path, frame)
            paths.append(out_path)
    cap.release()
    return paths


def _uniform_indices(total: int, n: int) -> List[int]:
    if total <= n:
        return list(range(total))
    step = total / n
    return [int(i * step) for i in range(n)]
