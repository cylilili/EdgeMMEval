# Qwen3-VL Profiling

This directory measures single-image inference for `Qwen/Qwen3-VL-2B-Instruct`.
It supports image classification and captioning on MPS, CUDA, or CPU, and records
latency, token counts, model metadata, and memory usage in JSON.

This is a performance baseline. It does not evaluate prediction accuracy.

## Setup

Python 3.12 is recommended.

```bash
cd /Volumes/thinkplus/MyEdgeMMEval/EdgeMMEval
python3.12 -m venv env-profile
source env-profile/bin/activate
python -m pip install -r profiling/requirements.txt
python profiling/single_image.py --check-env
```

The first model run may download several gigabytes from Hugging Face.

## Run the Four Experiments

The commands below measure ten images after one unmeasured warm-up request.

```bash
# Classification, 384 px
python profiling/single_image.py \
  --image-dir profiling/pics/coco --max-images 10 \
  --task classification --device mps \
  --max-image-side 384 --max-new-tokens 16 \
  --output profiling/results/coco_cls_res384.json

# Classification, original resolution
python profiling/single_image.py \
  --image-dir profiling/pics/coco --max-images 10 \
  --task classification --device mps \
  --max-image-side 0 --max-new-tokens 16 \
  --output profiling/results/coco_cls_resOrig.json

# Captioning, 384 px
python profiling/single_image.py \
  --image-dir profiling/pics/coco --max-images 10 \
  --task caption --device mps \
  --max-image-side 384 --max-new-tokens 64 \
  --output profiling/results/coco_cap_res384.json

# Captioning, original resolution
python profiling/single_image.py \
  --image-dir profiling/pics/coco --max-images 10 \
  --task caption --device mps \
  --max-image-side 0 --max-new-tokens 64 \
  --output profiling/results/coco_cap_resOrig.json
```

Use `--device cuda` or `--device cpu` for other backends. Individual files can
be supplied by repeating `--image PATH` instead of using `--image-dir`.

## Memory Measurement

CUDA uses PyTorch's native peak-memory counters. MPS does not expose equivalent
peak counters, so the script samples allocated and driver memory during each
request. The default interval is 10 ms and can be changed with:

```bash
--memory-sample-interval-ms 5
```

Each image records baseline, peak, request increase, and post-request memory.
MPS peak values are sampling-based estimates. CPU memory is not measured.

## Create the Summary Table

```bash
python profiling/summarize_results.py
```

This reads the 18 sweep JSON files in `profiling/results/sweep` and the four
sample-scaling JSON files in `profiling/results/stability`, writes
`profiling/results/summary_metrics.png`, and opens it in the system image
viewer. Use `--no-show` to save the image without opening it.

The report compares resolution scaling, MPS versus CPU, generation length, and
n=10 versus n=50 sample scaling and long-run memory stability.
Swap is used only for the resolution sweep because later experiments started
with pre-existing system swap that cannot be attributed to an individual run.

## Output Notes

- Warm-up requests are excluded from all reported statistics.
- Model loading time is stored separately from inference latency.
- Images are processed sequentially (batch size 1); the reported latency is per image.
- Image loading, RGB conversion, and optional resizing happen before the timer starts.
- `max_new_tokens` is an upper limit; actual generated token counts are recorded.
- P90 uses the nearest-rank method.
- MPS timing includes explicit synchronization around each measured request.
- Classification uses a fixed ten-label prompt on COCO images; these runs do not measure COCO accuracy.
- The recorded CPU runs use FP32 and MPS runs use FP16, so their speed difference includes precision.
