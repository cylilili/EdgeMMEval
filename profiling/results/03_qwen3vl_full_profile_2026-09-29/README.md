# Experiment 03 — Qwen3-VL Full Profiling Baseline

- Date: 2026-09-29
- Model: `Qwen/Qwen3-VL-2B-Instruct`
- Included groups: resolution scaling, generation length, MPS/CPU backend, and
  sample scaling/stability.
- Main result files: 18 JSON files in `sweep/` and four in `stability/`.
- Legacy system snapshots for non-resolution runs are retained for provenance;
  the current report attributes Swap only to the resolution group.

Regenerate the report from the repository root:

```bash
python profiling/summarize_results.py \
  --experiment-dir profiling/results/03_qwen3vl_full_profile_2026-09-29
```
