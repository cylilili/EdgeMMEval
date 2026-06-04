from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, List, Optional

from ..task import ModelConfig, Task
from ..metrics import word_error_rate


class ASRTask(Task):
    name = "asr"

    def __init__(self, data_dir: str, max_samples: int = 500):
        """
        data_dir: path to LibriSpeech test-clean root
                  (contains speaker subdirs with *.flac and *.txt files)
        """
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples

    def load_dataset(self) -> List[tuple]:
        """Returns list of (audio_path, normalised_reference_text)."""
        samples = []
        for trans_file in sorted(self.data_dir.rglob("*.trans.txt")):
            lines = trans_file.read_text().splitlines()
            audio_dir = trans_file.parent
            for line in lines:
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                utt_id, text = parts
                audio_path = audio_dir / f"{utt_id}.flac"
                if audio_path.exists():
                    samples.append((str(audio_path), _normalise(text)))
                if len(samples) >= self.max_samples:
                    return samples
        return samples

    def score(self, prediction: str, ground_truth: str) -> float:
        """1 - WER; clamped to [0, 1]."""
        wer = word_error_rate(ground_truth, _normalise(prediction))
        return max(0.0, 1.0 - wer)

    def classic_models(self) -> List[ModelConfig]:
        configs = []
        for size in ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"):
            size_c = size  # capture in closure
            configs.append(ModelConfig(
                name=f"whisper-{size_c}",
                kind="classic",
                loader=lambda s=size_c: _load_whisper(s),
                infer=_infer_whisper,
            ))
        configs.append(ModelConfig(
            name="vosk-small-en",
            kind="classic",
            loader=lambda: _load_vosk("vosk-model-small-en-us-0.15"),
            infer=_infer_vosk,
        ))
        configs.append(ModelConfig(
            name="vosk-en",
            kind="classic",
            loader=lambda: _load_vosk("vosk-model-en-us-0.22"),
            infer=_infer_vosk,
        ))
        return configs

    def mmllm_models(self) -> List[ModelConfig]:
        prompt = (
            "Transcribe the following audio to text. "
            "Reply with only the transcription, nothing else."
        )
        return [
            ModelConfig(
                name="gemma3n-e2b",
                kind="mmllm",
                loader=lambda: _ollama_handle("gemma3n:e2b"),
                infer=lambda model, path: _infer_ollama_audio(model, path, prompt),
            ),
            ModelConfig(
                name="gemma3n-e4b",
                kind="mmllm",
                loader=lambda: _ollama_handle("gemma3n:e4b"),
                infer=lambda model, path: _infer_ollama_audio(model, path, prompt),
            ),
        ]


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

def _load_whisper(size: str):
    import whisper
    return whisper.load_model(size)


def _infer_whisper(model, audio_path: str) -> str:
    result = model.transcribe(audio_path, language="en", fp16=False)
    return result["text"]


# --- Vosk -------------------------------------------------------------------

def _load_vosk(model_name: str):
    from vosk import Model, KaldiRecognizer
    import wave, json, os
    model_path = Path.home() / ".cache" / "vosk" / model_name
    if not model_path.exists():
        raise RuntimeError(
            f"Vosk model not found at {model_path}. "
            f"Download from https://alphacephei.com/vosk/models"
        )
    return Model(str(model_path))


def _infer_vosk(model, audio_path: str) -> str:
    from vosk import KaldiRecognizer
    import wave, json

    import soundfile as sf
    import io, struct
    data, sr = sf.read(audio_path, dtype="int16")
    rec = KaldiRecognizer(model, sr)
    rec.SetWords(False)
    chunk_size = 4000
    buf = data.tobytes()
    for i in range(0, len(buf), chunk_size):
        rec.AcceptWaveform(buf[i : i + chunk_size])
    result = json.loads(rec.FinalResult())
    return result.get("text", "")

def _ollama_handle(model_tag: str) -> str:
    import ollama
    available = {m["name"] for m in ollama.list()["models"]}
    if not any(model_tag in n for n in available):
        raise RuntimeError(
            f"Ollama model '{model_tag}' not found. Run: ollama pull {model_tag}"
        )
    return model_tag


def _infer_ollama_audio(model_tag: str, audio_path: str, prompt: str) -> str:
    """
    gemma3n supports audio input via Ollama.
    Encode as base64 and pass as 'audios' field.
    """
    import base64
    import ollama
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    resp = ollama.chat(
        model=model_tag,
        messages=[{
            "role": "user",
            "content": prompt,
            "audios": [audio_b64],
        }],
        options={"temperature": 0},
    )
    return resp["message"]["content"].strip()
