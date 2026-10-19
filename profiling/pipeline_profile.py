"""Measure single-image inference with a Qwen, Gemma, or SmolVLM2 model.

This records request-level latency and memory, not stage-level timing.
"""
import argparse
import importlib.metadata
import json
import math
import platform
import statistics
import threading
import time
from pathlib import Path


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
GEMMA_MODEL_ID = "google/gemma-3n-E2B-it"
SMOL_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
LABELS = (
    "tench, English springer, cassette player, chain saw, church, "
    "French horn, garbage truck, gas pump, golf ball, parachute"
)


def versions():
    result = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "transformers", "accelerate", "Pillow", "timm", "num2words"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--image", type=Path, action="append",
                        help="Image to measure; repeat this option for multiple images")
    parser.add_argument("--image-dir", type=Path,
                        help="Directory of JPG/PNG images to measure")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--task", choices=("classification", "caption"), default="caption")
    parser.add_argument("--model-id", choices=(DEFAULT_MODEL_ID, GEMMA_MODEL_ID, SMOL_MODEL_ID),
                        default=DEFAULT_MODEL_ID,
                        help="Hugging Face model to profile (default: Qwen3-VL-2B-Instruct)")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--max-image-side", type=int, default=0,
                        help="Resize the longest side before processing; 0 keeps original size")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached model files without downloading")
    parser.add_argument("--load-on-cpu-first", action="store_true",
                        help="Load weights on CPU, then move to MPS; avoids one large MPS allocator warm-up allocation")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Do not run one unmeasured warm-up request before measurements")
    parser.add_argument("--memory-sample-interval-ms", type=float, default=10.0,
                        help="MPS memory sampling interval in milliseconds (default: 10)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    environment = versions()
    if args.check_env:
        print(json.dumps(environment, indent=2))
        return
    image_paths = list(args.image or [])
    if args.image_dir:
        if not args.image_dir.is_dir():
            parser.error(f"Image directory does not exist: {args.image_dir}")
        image_paths.extend(sorted(
            p for p in args.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ))
    if not image_paths:
        parser.error("Provide --image PATH (repeatable) or --image-dir PATH")
    missing = [str(p) for p in image_paths if not p.is_file()]
    if missing:
        parser.error(f"Image does not exist: {missing[0]}")
    if args.max_images < 1:
        parser.error("--max-images must be positive")
    image_paths = image_paths[:args.max_images]
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.max_image_side < 0:
        parser.error("--max-image-side must be nonnegative")
    if args.memory_sample_interval_ms <= 0:
        parser.error("--memory-sample-interval-ms must be positive")

    import torch
    from PIL import Image
    from transformers import (
        AutoProcessor,
        Gemma3nForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
        SmolVLMForConditionalGeneration,
    )

    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this Python/PyTorch environment")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python/PyTorch environment")
    if args.load_on_cpu_first and device != "mps":
        parser.error("--load-on-cpu-first requires --device mps")
    dtype = (torch.float32 if device == "cpu" else torch.float16) if args.dtype == "auto" else getattr(torch, args.dtype)

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    class ModuleTimer:
        def __init__(self):
            self.reset()

        def reset(self):
            self.started_at = None
            self.elapsed_seconds = 0.0
            self.call_count = 0
            self.durations_seconds = []

        def before_forward(self, module, args):
            synchronize()
            self.started_at = time.perf_counter()

        def after_forward(self, module, args, output):
            synchronize()

            if self.started_at is None:
                raise RuntimeError("ModuleTimer ended without starting")

            duration = time.perf_counter() - self.started_at
            self.elapsed_seconds += duration
            self.durations_seconds.append(duration)
            self.call_count += 1
            self.started_at = None

    print(f"Loading {args.model_id} on {device} ({dtype}). First use downloads weights.", flush=True)
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    model_class = {
        DEFAULT_MODEL_ID: Qwen3VLForConditionalGeneration,
        GEMMA_MODEL_ID: Gemma3nForConditionalGeneration,
        SMOL_MODEL_ID: SmolVLMForConditionalGeneration,
    }[args.model_id]
    load_device = "cpu" if args.load_on_cpu_first else device
    model = model_class.from_pretrained(
        args.model_id, dtype=dtype, device_map={"": load_device},
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    )
    if args.load_on_cpu_first:
        print("Moving model from CPU to MPS; this can use substantial unified memory.", flush=True)
        model = model.to(device)
    model.eval()


    if args.model_id != DEFAULT_MODEL_ID:
        parser.error(
            "Pipeline stage profiling currently supports only "
            "Qwen/Qwen3-VL-2B-Instruct"
        )

    visual_timer = ModuleTimer()

    visual_pre_hook = model.model.visual.register_forward_pre_hook(
        visual_timer.before_forward
    )

    visual_post_hook = model.model.visual.register_forward_hook(
        visual_timer.after_forward
    )


    merger_timer = ModuleTimer()

    merger_modules = [
        model.model.visual.merger,
        *model.model.visual.deepstack_merger_list,
    ]

    merger_hook_handles = []

    for merger_module in merger_modules:
        merger_hook_handles.extend([
            merger_module.register_forward_pre_hook(
                merger_timer.before_forward
            ),
            merger_module.register_forward_hook(
                merger_timer.after_forward
            ),
        ])

    expected_merger_calls = len(merger_modules)


    model_forward_timer = ModuleTimer()

    model_forward_pre_hook = model.register_forward_pre_hook(
        model_forward_timer.before_forward
    )

    model_forward_post_hook = model.register_forward_hook(
        model_forward_timer.after_forward
    )


    language_model_timer = ModuleTimer()

    language_model_pre_hook = (
        model.model.language_model.register_forward_pre_hook(
            language_model_timer.before_forward
        )
    )

    language_model_post_hook = (
        model.model.language_model.register_forward_hook(
            language_model_timer.after_forward
        )
    )


    lm_head_timer = ModuleTimer()

    lm_head_pre_hook = model.lm_head.register_forward_pre_hook(
        lm_head_timer.before_forward
    )

    lm_head_post_hook = model.lm_head.register_forward_hook(
        lm_head_timer.after_forward
    )


    synchronize()
    load_s = time.perf_counter() - start
    print(f"Model loaded in {load_s:.1f}s. Preparing image...", flush=True)

    prompt = (
        f"Classify this image into exactly one of: {LABELS}. "
        "Return only the category name."
        if args.task == "classification" else
        "Describe the visible objects and their spatial relationships in this image "
        "in two or three sentences. Do not speculate about things that are not visible."
    )
    class MPSMemorySampler:
        """Estimate per-request MPS peak memory by periodic sampling."""

        def __init__(self, interval_seconds):
            self.interval_seconds = interval_seconds
            self.stop_event = threading.Event()
            self.thread = None
            self.baseline_allocated_mb = 0.0
            self.baseline_driver_mb = 0.0
            self.peak_allocated_mb = 0.0
            self.peak_driver_mb = 0.0

        @staticmethod
        def read():
            return (
                torch.mps.current_allocated_memory() / 2**20,
                torch.mps.driver_allocated_memory() / 2**20,
            )

        def sample(self):
            allocated_mb, driver_mb = self.read()
            self.peak_allocated_mb = max(self.peak_allocated_mb, allocated_mb)
            self.peak_driver_mb = max(self.peak_driver_mb, driver_mb)

        def run(self):
            while not self.stop_event.wait(self.interval_seconds):
                self.sample()

        def start(self):
            synchronize()
            self.baseline_allocated_mb, self.baseline_driver_mb = self.read()
            self.peak_allocated_mb = self.baseline_allocated_mb
            self.peak_driver_mb = self.baseline_driver_mb
            self.stop_event.clear()
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

        def stop(self):
            synchronize()
            self.sample()
            self.stop_event.set()
            if self.thread is not None:
                self.thread.join()
            after_allocated_mb, after_driver_mb = self.read()
            return {
                "sampling_interval_ms": self.interval_seconds * 1000,
                "baseline_allocated_mb": self.baseline_allocated_mb,
                "baseline_driver_allocated_mb": self.baseline_driver_mb,
                "peak_allocated_mb": self.peak_allocated_mb,
                "peak_driver_allocated_mb": self.peak_driver_mb,
                "peak_increment_allocated_mb": (
                    self.peak_allocated_mb - self.baseline_allocated_mb
                ),
                "peak_increment_driver_mb": (
                    self.peak_driver_mb - self.baseline_driver_mb
                ),
                "after_allocated_mb": after_allocated_mb,
                "after_driver_allocated_mb": after_driver_mb,
                "recommended_max_memory_mb": torch.mps.recommended_max_memory() / 2**20,
                "peak_is_sampled": True,
            }

    def start_memory_tracking():
        if device == "mps":
            tracker = MPSMemorySampler(args.memory_sample_interval_ms / 1000)
            tracker.start()
            return tracker
        if device == "cuda":
            synchronize()
            torch.cuda.reset_peak_memory_stats()
            return {
                "baseline_allocated_mb": torch.cuda.memory_allocated() / 2**20,
                "baseline_reserved_mb": torch.cuda.memory_reserved() / 2**20,
            }
        return None

    def stop_memory_tracking(tracker):
        if device == "mps":
            return tracker.stop()
        if device == "cuda":
            synchronize()
            after_allocated_mb = torch.cuda.memory_allocated() / 2**20
            after_reserved_mb = torch.cuda.memory_reserved() / 2**20
            peak_allocated_mb = torch.cuda.max_memory_allocated() / 2**20
            peak_reserved_mb = torch.cuda.max_memory_reserved() / 2**20
            return {
                **tracker,
                "peak_allocated_mb": peak_allocated_mb,
                "peak_reserved_mb": peak_reserved_mb,
                "peak_increment_allocated_mb": (
                    peak_allocated_mb - tracker["baseline_allocated_mb"]
                ),
                "peak_increment_reserved_mb": (
                    peak_reserved_mb - tracker["baseline_reserved_mb"]
                ),
                "after_allocated_mb": after_allocated_mb,
                "after_reserved_mb": after_reserved_mb,
                "peak_is_sampled": False,
            }
        return {}

    def run_one(path: Path, measured: bool):
        synchronize()
        memory_tracker = start_memory_tracking()
        total_start = time.perf_counter()
        preprocessing_start = total_start

        try:
            with Image.open(path) as source:
                picture = source.convert("RGB")
            original_size = picture.size
            if args.max_image_side:
                picture.thumbnail((args.max_image_side, args.max_image_side), Image.Resampling.LANCZOS)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": picture},
                {"type": "text", "text": prompt},
            ]}]

            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

            synchronize()
            preprocessing_s = time.perf_counter() - preprocessing_start

            visual_timer.reset()
            merger_timer.reset()
            model_forward_timer.reset()
            language_model_timer.reset()
            lm_head_timer.reset()

            input_tokens = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

            visual_module_s = visual_timer.elapsed_seconds
            visual_module_calls = visual_timer.call_count

            merger_s = merger_timer.elapsed_seconds
            merger_calls = merger_timer.call_count

            model_forward_calls = model_forward_timer.call_count
            model_forward_durations = model_forward_timer.durations_seconds

            language_model_calls = language_model_timer.call_count
            language_model_durations = (
                language_model_timer.durations_seconds
            )

            lm_head_calls = lm_head_timer.call_count
            lm_head_durations = lm_head_timer.durations_seconds

            if visual_module_calls != 1:
                raise RuntimeError(
                    f"Expected one visual forward call, got {visual_module_calls}"
                )
            if visual_module_s <= 0:
                raise RuntimeError("Visual module latency must be positive")

            if merger_calls != expected_merger_calls:
                raise RuntimeError(
                    f"Expected {expected_merger_calls} merger calls, "
                    f"got {merger_calls}"
                )

            if merger_s <= 0:
                raise RuntimeError("Merger latency must be positive")

            if merger_s >= visual_module_s:
                raise RuntimeError(
                    "Merger latency must be smaller than total visual latency"
                )

            vision_encoder_s = visual_module_s - merger_s

            if model_forward_calls < 1:
                raise RuntimeError("Expected at least one model forward call")

            if len(model_forward_durations) != model_forward_calls:
                raise RuntimeError(
                    "Model forward duration count does not match call count"
                )

            first_forward_s = model_forward_durations[0]

            if first_forward_s <= 0:
                raise RuntimeError("First model forward latency must be positive")

            if language_model_calls != model_forward_calls:
                raise RuntimeError(
                    f"Language model calls ({language_model_calls}) "
                    f"do not match model forward calls "
                    f"({model_forward_calls})"
                )

            if len(language_model_durations) != language_model_calls:
                raise RuntimeError(
                    "Language model duration count does not match call count"
                )

            first_language_model_s = language_model_durations[0]

            if first_language_model_s <= 0:
                raise RuntimeError(
                    "First language model latency must be positive"
                )

            if first_language_model_s >= first_forward_s:
                raise RuntimeError(
                    "Language model latency must be smaller than "
                    "the complete first forward latency"
                )

            if lm_head_calls != model_forward_calls:
                raise RuntimeError(
                    f"LM head calls ({lm_head_calls}) do not match "
                    f"model forward calls ({model_forward_calls})"
                )

            if len(lm_head_durations) != lm_head_calls:
                raise RuntimeError(
                    "LM head duration count does not match call count"
                )

            first_lm_head_s = lm_head_durations[0]

            if first_lm_head_s <= 0:
                raise RuntimeError(
                    "First LM head latency must be positive"
                )

            llm_prefill_s = (
                first_language_model_s
                + first_lm_head_s
            )

            if llm_prefill_s >= first_forward_s:
              raise RuntimeError(
                  "LLM prefill must be smaller than "
                  "the complete first forward"
              )

            generated = output[0, input_tokens:]
            answer = processor.decode(generated, skip_special_tokens=True).strip()
            empty_answer_diagnostics = {}
            if not answer:
                empty_answer_diagnostics = {
                    "raw_generated_text": processor.decode(generated, skip_special_tokens=False),
                    "generated_token_ids": generated.tolist(),
                }
                print("  Warning: generated tokens decoded to an empty answer; raw tokens saved in JSON.", flush=True)

            synchronize()
            elapsed_s = time.perf_counter() - total_start
        finally:
            memory = stop_memory_tracking(memory_tracker)
        memory_after = {
            key.removeprefix("after_"): value
            for key, value in memory.items()
            if key.startswith("after_")
        }
        return {
            "image": str(path.resolve()), "image_size": picture.size,
            "original_image_size": original_size,
            "input_tokens_including_visual_placeholders": input_tokens,
            "generated_tokens_including_special_tokens": generated.numel(),
            "latency_seconds": elapsed_s,
            "answer": answer,
            "stages_seconds": {
                "image_preprocessing": preprocessing_s,
                "vision_encoder": vision_encoder_s,
                "multimodal_projection_mergers": merger_s,
                "llm_prefill": llm_prefill_s,
            },
            "stage_diagnostics": {
                "visual_module_total_seconds": visual_module_s,
                "visual_module_calls": visual_module_calls,
                "merger_calls": merger_calls,
                "expected_merger_calls": expected_merger_calls,
                "model_forward_calls": model_forward_calls,
                "model_forward_durations_seconds": model_forward_durations,
                "first_forward_seconds": first_forward_s,
                "language_model_calls": language_model_calls,
                "language_model_durations_seconds": language_model_durations,
                "first_language_model_seconds": first_language_model_s,
                "lm_head_calls": lm_head_calls,
                "lm_head_durations_seconds": lm_head_durations,
                "first_lm_head_seconds": first_lm_head_s,
            },
            **empty_answer_diagnostics,
            "memory": memory, "memory_after": memory_after,
            "measured": measured,
        }

    if not args.no_warmup:
        print(f"Warm-up: {image_paths[0]}", flush=True)
        run_one(image_paths[0], measured=False)

    per_image = []
    for i, path in enumerate(image_paths, 1):
        print(f"Measuring image {i}/{len(image_paths)}: {path}", flush=True)
        item = run_one(path, measured=True)
        per_image.append(item)
        memory_text = ""
        if "peak_driver_allocated_mb" in item["memory"]:
            memory_text = f", peak_driver_memory={item['memory']['peak_driver_allocated_mb']:.1f}MB"
        elif "peak_allocated_mb" in item["memory"]:
            memory_text = f", peak_memory={item['memory']['peak_allocated_mb']:.1f}MB"
        print(
            f"  latency={item['latency_seconds']:.3f}s, "
            f"generated_tokens={item['generated_tokens_including_special_tokens']}"
            f"{memory_text}",
            flush=True,
        )

    latencies = [x["latency_seconds"] for x in per_image]
    summary = {
        "count": len(latencies),
        "mean_seconds": statistics.mean(latencies),
        "median_seconds": statistics.median(latencies),
        "p90_seconds": sorted(latencies)[math.ceil(0.9 * len(latencies)) - 1],
        "stdev_seconds": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min_seconds": min(latencies), "max_seconds": max(latencies),
    }
    memory_fields = (
        "peak_allocated_mb",
        "peak_driver_allocated_mb",
        "peak_reserved_mb",
        "peak_increment_allocated_mb",
        "peak_increment_driver_mb",
        "peak_increment_reserved_mb",
    )
    memory_summary = {}
    for field in memory_fields:
        values = [
            item["memory"][field]
            for item in per_image
            if field in item["memory"]
        ]
        if values:
            memory_summary[field] = {
                "mean_mb": statistics.mean(values),
                "max_mb": max(values),
            }
    if memory_summary:
        summary["memory"] = memory_summary
    record = {
        "purpose": "multi-image baseline; warm-up excluded from measured statistics",
        "model": args.model_id, "task": args.task, "device": device,
        "dtype": str(dtype), "attention_backend": "eager",
        "load_on_cpu_first": args.load_on_cpu_first,
        "prompt": prompt, "max_new_tokens": args.max_new_tokens,
        "max_image_side": args.max_image_side,
        "load_seconds_including_download_if_needed": load_s,
        "warmup": not args.no_warmup, "summary": summary,
        "images": per_image, "environment": environment,
    }
    print(json.dumps(record, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
