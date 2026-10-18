# Experiment 07 — SmolVLM2 Full Profiling

- Date: 2026-10-18
- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Included groups: resolution scaling and generation length.
- Excluded groups: sample scaling/stability and MPS/CPU backend comparison.
- Main result files: 14 JSON files in `sweep/`.

Regenerate the report from the repository root:

```bash
python profiling/summarize_results.py \
  --experiment-dir profiling/results/07_smolvlm2_full_profile_2026-10-18
```
