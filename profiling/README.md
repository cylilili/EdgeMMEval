# Vision-Language Model Profiling

This directory measures single-image inference latency, token counts, and
accelerator memory for supported vision-language models. It can run resolution,
generation-length, sample-stability, and optional hardware-backend experiments.
It does not measure prediction accuracy. Images are processed sequentially with
batch size 1, after one unmeasured warm-up request.

## Directory Structure

```text
profiling/
├── README.md
├── requirements.txt
├── single_image.py             # Runs one profiling configuration
├── summarize_results.py        # Builds an English PNG report
├── pics/coco/                  # Input images, sorted by filename
└── results/
    ├── README.md               # Experiment index
    ├── 01_.../                 # One complete experiment per folder
    ├── 02_.../
    └── NN_experiment_YYYY-MM-DD/
        ├── README.md
        ├── sweep/
        ├── stability/          # Optional
        └── summary_metrics.png
```

Numbered result directories record experiment order. A model name may appear in
the folder name for clarity, but results are organized by experiment, not by
model. See [`results/README.md`](results/README.md) for the current inventory.

## 1. Prepare the Environment

Run all commands from the repository root:

```bash
cd /Volumes/thinkplus/MyEdgeMMEval/EdgeMMEval
source env-profile312/bin/activate
python --version
python -m pip install -r profiling/requirements.txt
python profiling/single_image.py --check-env
```

Python 3.12 is expected. Supported model IDs are:

| Model | `MODEL_ID` |
| --- | --- |
| Qwen3-VL-2B | `Qwen/Qwen3-VL-2B-Instruct` |
| SmolVLM2-2.2B | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` |
| Gemma 3n E2B | `google/gemma-3n-E2B-it` |

Check the Hugging Face cache used by the active terminal:

```bash
python -c 'from huggingface_hub import constants; print(constants.HF_HUB_CACHE)'
```

The first run downloads missing model files. To download explicitly, use:

```bash
hf download Qwen/Qwen3-VL-2B-Instruct
```

For a gated model such as Gemma, accept its license on Hugging Face and run
`hf auth login` first. Add `--local-files-only` to profiling commands only
after the complete checkpoint is present in the active cache.

## 2. Create One Experiment Directory

Inspect the existing sequence and choose the next unused number:

```bash
ls -1 profiling/results
```

Set the experiment and model for the current terminal. This example creates the
next Qwen repeat:

```bash
EXPERIMENT_ID="08_qwen3vl_repeat_2026-10-18"
EXPERIMENT_DIR="profiling/results/${EXPERIMENT_ID}"
MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"

mkdir -p "$EXPERIMENT_DIR/sweep" "$EXPERIMENT_DIR/stability"
```

Use lowercase names, a two-digit sequence number, and an ISO date. Variables
defined above exist only in the current terminal. If the terminal is closed,
activate the environment and define them again before continuing.

Do not reuse an existing experiment directory for a new repeat. This prevents
old and new JSON files from being mixed in one report.

## 3. Run a One-Image Pilot

Run a pilot before a new model or a memory-intensive configuration:

```bash
python profiling/single_image.py \
  --model-id "$MODEL_ID" \
  --image-dir profiling/pics/coco \
  --max-images 1 \
  --task caption \
  --device mps \
  --dtype float16 \
  --max-image-side 224 \
  --max-new-tokens 64 \
  --memory-sample-interval-ms 10 \
  --output "$EXPERIMENT_DIR/pilot_caption.json"
```

Confirm that the command completes and that `answer` is nonempty in the JSON.
Pilot files stay at the experiment root, so they are not included in the formal
summary report.

## 4. Resolution Scaling

This group runs classification and captioning at 224, 384, 512, 768, and
original resolution. It is the only group that records system Swap and memory
pressure. Swap is a system-wide cumulative value, so begin from a known low-Swap
state if the goal is to identify a pressure threshold.

Check current Swap first:

```bash
/usr/sbin/sysctl vm.swapusage
```

Define this helper in the current terminal:

```bash
capture_resolution_run() {
  local output="$1"
  shift
  local prefix="${output%.json}"

  /usr/sbin/sysctl vm.swapusage > "${prefix}_swap_before.txt"
  /usr/bin/memory_pressure -Q > "${prefix}_pressure_before.txt"

  python profiling/single_image.py "$@" --output "$output"
  local run_exit=$?

  /usr/sbin/sysctl vm.swapusage > "${prefix}_swap_after.txt"
  /usr/bin/memory_pressure -Q > "${prefix}_pressure_after.txt"

  return $run_exit
}
```

Run all ten configurations:

```bash
for resolution in 224 384 512 768 0; do
  if [[ "$resolution" == "0" ]]; then
    tag="original"
  else
    tag="$resolution"
  fi

  capture_resolution_run \
    "$EXPERIMENT_DIR/sweep/resolution_cls_${tag}.json" \
    --model-id "$MODEL_ID" \
    --image-dir profiling/pics/coco \
    --max-images 10 \
    --task classification \
    --device mps \
    --max-image-side "$resolution" \
    --max-new-tokens 16 \
    --memory-sample-interval-ms 10

  capture_resolution_run \
    "$EXPERIMENT_DIR/sweep/resolution_cap_${tag}.json" \
    --model-id "$MODEL_ID" \
    --image-dir profiling/pics/coco \
    --max-images 10 \
    --task caption \
    --device mps \
    --max-image-side "$resolution" \
    --max-new-tokens 64 \
    --memory-sample-interval-ms 10
done
```

If a larger setting may exhaust memory, test that setting with `--max-images 1`
and a separate pilot filename before running the ten-image configuration.

## 5. Generation Length

Captioning is used because classification normally stops after only a few
tokens. The report fits latency against the actual generated-token count.

```bash
for tokens in 16 32 64 128; do
  python profiling/single_image.py \
    --model-id "$MODEL_ID" \
    --image-dir profiling/pics/coco \
    --max-images 10 \
    --task caption \
    --device mps \
    --max-image-side 384 \
    --max-new-tokens "$tokens" \
    --memory-sample-interval-ms 10 \
    --output "$EXPERIMENT_DIR/sweep/generation_cap_${tokens}.json"
done
```

## 6. Sample Scaling and Stability

The image directory must contain at least 50 images. Both sample sizes use
original resolution. Swap is intentionally not collected for this group.

```bash
for n in 10 50; do
  python profiling/single_image.py \
    --model-id "$MODEL_ID" \
    --image-dir profiling/pics/coco \
    --max-images "$n" \
    --task classification \
    --device mps \
    --max-image-side 0 \
    --max-new-tokens 16 \
    --memory-sample-interval-ms 10 \
    --output "$EXPERIMENT_DIR/stability/classification_n${n}.json"

  python profiling/single_image.py \
    --model-id "$MODEL_ID" \
    --image-dir profiling/pics/coco \
    --max-images "$n" \
    --task caption \
    --device mps \
    --max-image-side 0 \
    --max-new-tokens 64 \
    --memory-sample-interval-ms 10 \
    --output "$EXPERIMENT_DIR/stability/caption_n${n}.json"
done
```

This group is optional. If it is omitted, the report simply leaves out the
sample-scaling table.

## 7. Hardware Backend Comparison (Optional)

Run this group only when a CPU-versus-MPS comparison is needed. Swap is not
collected or attributed to these runs.

```bash
for device in mps cpu; do
  python profiling/single_image.py \
    --model-id "$MODEL_ID" \
    --image-dir profiling/pics/coco \
    --max-images 10 \
    --task classification \
    --device "$device" \
    --max-image-side 384 \
    --max-new-tokens 16 \
    --memory-sample-interval-ms 10 \
    --output "$EXPERIMENT_DIR/sweep/hardware_cls_${device}.json"

  python profiling/single_image.py \
    --model-id "$MODEL_ID" \
    --image-dir profiling/pics/coco \
    --max-images 10 \
    --task caption \
    --device "$device" \
    --max-image-side 384 \
    --max-new-tokens 64 \
    --memory-sample-interval-ms 10 \
    --output "$EXPERIMENT_DIR/sweep/hardware_cap_${device}.json"
done
```

With `--dtype auto`, MPS uses FP16 and CPU uses FP32. Therefore, default
speedup includes both backend and precision differences. Use `--dtype float32`
on both backends only when a precision-controlled comparison is required.

## 8. Generate the PNG Report

Generate the report for one experiment directory:

```bash
python profiling/summarize_results.py \
  --experiment-dir "$EXPERIMENT_DIR"
```

The script reads `sweep/*.json` and, when present, `stability/*.json`. It
writes `summary_metrics.png` into the same experiment directory and opens the
image in the system viewer. Use `--no-show` to save without opening it:

```bash
python profiling/summarize_results.py \
  --experiment-dir "$EXPERIMENT_DIR" \
  --no-show
```

Only available groups are rendered. Hardware and stability results are not
required. All JSON files included in one report must use the same model.

The older explicit-directory interface remains available:

```bash
python profiling/summarize_results.py \
  --input-dir "$EXPERIMENT_DIR/sweep" \
  --stability-dir "$EXPERIMENT_DIR/stability" \
  --output "$EXPERIMENT_DIR/summary_metrics.png"
```

## 9. Document the Experiment

Add a short `README.md` inside the experiment directory. Record:

- date and model ID;
- device and dtype;
- included and intentionally skipped groups;
- sample source and sample counts;
- failures, empty outputs, or out-of-memory settings;
- the report regeneration command.

Then add one row to [`results/README.md`](results/README.md).

## 10. Repeatability Rules

For a direct repeat, keep these fixed:

- model ID, prompt, image directory, and sorted image order;
- `--max-images`, `--max-image-side`, and `--max-new-tokens`;
- backend, dtype behavior, and 10 ms MPS sampling interval;
- one unmeasured warm-up and batch size 1;
- power connection and background application load where possible;
- sequential execution; do not run experiments in parallel.

Use a new numbered experiment directory for every repeat. For a Swap-threshold
repeat, restart macOS first if Swap is already large. A 768 px limit is
equivalent to original resolution when every source image has a longest side
below 768 px.

## 11. Measurement Notes

- Image loading, RGB conversion, and optional resizing happen before timing.
- Preprocessing, device transfer, generation, decoding, and synchronization are
  included in per-image latency.
- Model loading time is stored separately.
- Warm-up requests are excluded from reported statistics.
- `max_new_tokens` is an upper limit; actual generated tokens are recorded.
- P90 and P99 use the nearest-rank method.
- CUDA uses native peak-memory counters.
- MPS peak memory is estimated by periodic sampling because PyTorch MPS does
  not expose an equivalent native peak counter.
- Post-request allocated-memory trends help detect live-allocation growth.
  Metal driver-memory growth can also reflect reusable allocator pools and does
  not by itself establish a leak.
- Swap is system-wide. A positive before/after change is attributable only when
  the experiment begins from a controlled low-Swap state.
- Classification uses a fixed ten-label prompt on COCO images and does not
  measure COCO classification accuracy.

## 12. Model-Specific Notes

### Qwen3-VL-2B

This is the default model and the reference configuration for the commands
above.

### SmolVLM2-2.2B

The checkpoint is public but large. Its processor requires `num2words`, which
is included in `profiling/requirements.txt`. The script uses eager attention;
`flash-attn` is not required. On a 16 GB Mac, run a one-image pilot before
each larger resolution. An unsupported setting is a feasibility result and
should not be replaced silently with a different dtype.

### Gemma 3n E2B

The Hugging Face checkpoint is separate from Ollama weights and requires model
license acceptance. If direct MPS loading fails, `--load-on-cpu-first` loads
on CPU before moving to MPS. This can still require substantial unified memory.
The 2026-10-17 pilots in Experiment 05 generated empty pad-only output under the
tested configuration; they are diagnostics, not valid caption benchmarks.
