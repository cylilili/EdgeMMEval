# Profiling Experiment Index

Each numbered directory is one experiment or one clearly scoped validation run.
The number records execution order; model names are descriptive metadata, not
the primary directory hierarchy.

| No. | Experiment | Date | Model | Contents | Report |
| ---: | --- | --- | --- | --- | --- |
| 01 | [Setup validation](01_qwen3vl_setup_validation_2026-09-26/) | 2026-09-26 | Qwen3-VL-2B | CPU diagnostic, first caption, 5- and 10-image baselines | No |
| 02 | [Four-configuration baseline](02_qwen3vl_four_config_2026-09-27/) | 2026-09-27 | Qwen3-VL-2B | Classification/caption at 384 px and original size, repeated twice | No |
| 03 | [Full profiling baseline](03_qwen3vl_full_profile_2026-09-29/) | 2026-09-29 | Qwen3-VL-2B | Resolution, generation length, hardware backend, stability | [PNG](03_qwen3vl_full_profile_2026-09-29/summary_metrics.png) |
| 04 | [Repeat validation](04_qwen3vl_repeat_2026-10-17/) | 2026-10-17 | Qwen3-VL-2B | Resolution, generation length, stability | [PNG](04_qwen3vl_repeat_2026-10-17/summary_metrics.png) |
| 05 | [Gemma 3n pilot](05_gemma3n_pilot_2026-10-17/) | 2026-10-17 | Gemma 3n E2B | Pilot and empty-output diagnostic | No |
| 06 | [SmolVLM2 pilot](06_smolvlm2_pilot_2026-10-18/) | 2026-10-18 | SmolVLM2-2.2B | One-image feasibility pilot | No |
| 07 | [Full SmolVLM2 profiling](07_smolvlm2_full_profile_2026-10-18/) | 2026-10-18 | SmolVLM2-2.2B | Resolution and generation length | [PNG](07_smolvlm2_full_profile_2026-10-18/summary_metrics.png) |

Formal profiling experiments use this layout:

```text
NN_experiment_name_YYYY-MM-DD/
├── README.md
├── sweep/
├── stability/              # Optional
└── summary_metrics.png     # Generated report
```

The report generator renders only groups backed by available JSON files. A
missing optional group does not prevent report generation.

Create the next experiment with the next unused number. Full commands and
naming rules are in [`../README.md`](../README.md).
