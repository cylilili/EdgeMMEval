# Experiment 04 — Qwen3-VL Repeat Validation

- Date: 2026-10-17
- Model: `Qwen/Qwen3-VL-2B-Instruct`
- Included groups: resolution scaling, generation length, and sample
  scaling/stability.
- Excluded group: MPS/CPU backend comparison.
- Main result files: 14 JSON files in `sweep/` and four in `stability/`.

Regenerate the report from the repository root:

```bash
python profiling/summarize_results.py \
  --experiment-dir profiling/results/04_qwen3vl_repeat_2026-10-17
```
