# EdgeMMEval

## Package Structure

```
edgemmeval/
├── README.md                   
├── requirements.txt            
├── setup.py                    
├── run_benchmark.py            
├── edgemmeval/
│   ├── __init__.py             
│   ├── harness.py              
│   ├── task.py                 
│   ├── metrics.py              
│   ├── deployability.py        
│   ├── memory.py               
│   ├── report.py               
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── image_classification.py
│   │   ├── asr.py
│   │   ├── object_counting.py
│   │   ├── action_recognition.py
│   │   ├── hazard_detection.py
│   │   └── ocr.py              
│   └── scripts/
│       ├── generate_figures.py      
│       └── generate_harness_demo.py 
└── examples/
    ├── quick_ic.py
    ├── quick_asr.py
    ├── quick_hd.py
    └── quick_all.py
```

## Installation

From the `edgemmeval/` directory (this directory):

```bash
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

Python 3.9 or newer is required.

---

## Prerequisites

### Ollama (required for MMLLM models)

Install Ollama from https://ollama.com and start the server:

```bash
ollama serve
```

Then pull the models used in the benchmark:

```bash
ollama pull gemma3n:e2b
ollama pull gemma3n:e4b
ollama pull qwen2.5vl:3b
ollama pull mistral:3b-instruct
```

For the OCR task:

```bash
ollama pull gemma3:4b
ollama pull qwen3-vl:2b
```

### HuggingFace token (for BLIP-2 / gated models)

Some models (e.g. BLIP-2, MedGemma) are hosted on HuggingFace and may require authentication:

```bash
huggingface-cli login
```

Or set the environment variable:

```bash
export HF_TOKEN=hf_...
```

### Vosk models (for ASR task)

Download Vosk models to `~/.cache/vosk/`:

```bash
mkdir -p ~/.cache/vosk
# Small English model (~40 MB)
wget -qO- https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip | \
    unzip - -d ~/.cache/vosk/
# Large English model (~1.8 GB)
wget -qO- https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip | \
    unzip - -d ~/.cache/vosk/
```

---

## Quick Start

Run a single task with a small sample budget to verify the setup:

```bash
# Image Classification (Imagenette, 10 samples)
python examples/quick_ic.py --data /path/to/imagenette2 --n-samples 10 --no-deploy-probe

# Automatic Speech Recognition (LibriSpeech test-clean, 10 samples)
python examples/quick_asr.py --data /path/to/LibriSpeech/test-clean --n-samples 10

# Hazard Detection (DetectiumFire, 10 videos)
python examples/quick_hd.py --data /path/to/detectium_fire --n-samples 10

# All five tasks (5 samples each)
python examples/quick_all.py \
    --data-ic /path/to/imagenette2 \
    --data-asr /path/to/LibriSpeech/test-clean \
    --data-oc-images /path/to/coco/val2017 \
    --data-oc-ann /path/to/coco/annotations/instances_val2017.json \
    --data-ar /path/to/kinetics/mini_val \
    --data-hd /path/to/detectium_fire \
    --n-samples 5
```

Add `--output results.json` to any command to save structured results.

---

## Full CLI Usage (`run_benchmark.py`)

```
python run_benchmark.py [--all | --task TASK] [OPTIONS]
```

### Task selection

| Flag | Description |
|---|---|
| `--all` | Run all five tasks |
| `--task ic` | Image classification (Imagenette) |
| `--task asr` | Automatic speech recognition (LibriSpeech) |
| `--task oc` | Object counting (COCO) |
| `--task ar` | Action recognition (Kinetics mini-val) |
| `--task hd` | Hazard detection (DetectiumFire) |

### Dataset paths

| Flag | Default | Description |
|---|---|---|
| `--data-ic PATH` | `/data/imagenette2` | Imagenette root |
| `--data-asr PATH` | `/data/librispeech/test-clean` | LibriSpeech test-clean root |
| `--data-oc-images PATH` | `/data/coco/val2017` | COCO val2017 images directory |
| `--data-oc-ann PATH` | `/data/coco/annotations/instances_val2017.json` | COCO annotation JSON |
| `--data-ar PATH` | `/data/kinetics/mini_val` | Kinetics mini-val root |
| `--data-hd PATH` | `/data/detectium_fire` | DetectiumFire root |

### Sample budget

| Flag | Description |
|---|---|
| `--budget quick` | 50 samples per task (smoke-test, ~minutes) |
| `--budget standard` | 500 samples per task (~1 hour) |
| `--budget full` | All samples (publication quality) |
| `--n-samples N` | Exact sample cap; overrides `--budget` |

### Other options

| Flag | Description |
|---|---|
| `--no-deploy-probe` | Skip the deployability probe pass (faster iteration) |
| `--hardware-id STR` | Override the hardware label shown in reports |
| `--oc-class CLASS` | COCO class to count (default: `person`) |
| `--hd-k N` | Periodic hybrid invocation interval (default: `5`) |
| `--output FILE.json` | Save structured results to JSON |
| `--quiet` | Suppress per-model progress output |

### Example

```bash
python run_benchmark.py --task ic \
    --data-ic /data/imagenette2 \
    --budget standard \
    --output ic_results.json
```

---

## Extending with a New Task

All tasks implement the four-method `Task` ABC from `edgemmeval/task.py`:

```python
from edgemmeval.task import Task, ModelConfig

class MyNewTask(Task):
    name = "my_task"

    def load_dataset(self):
        # Return list of (input, ground_truth) pairs
        return [(input1, gt1), (input2, gt2), ...]

    def score(self, prediction, ground_truth):
        # Return float in [0, 1]; higher is better
        return 1.0 if prediction == ground_truth else 0.0

    def classic_models(self):
        return [
            ModelConfig(
                name="my_classic_model",
                kind="classic",
                loader=lambda: load_my_model(),
                infer=lambda model, inp: my_model_predict(model, inp),
            )
        ]

    def mmllm_models(self):
        return [
            ModelConfig(
                name="gemma3n-e2b",
                kind="mmllm",
                loader=lambda: "gemma3n:e2b",  # Ollama tag as handle
                infer=lambda model, inp: call_ollama(model, inp),
            )
        ]
```

Then run it through the harness:

```python
from edgemmeval import BenchmarkEngine, report

engine = BenchmarkEngine()
run = engine.run(MyNewTask("/path/to/data"), n_samples=100)
report.print_summary([run])
report.save_json([run], "my_task_results.json")
```

No harness internals need to be modified. The OCR task (`edgemmeval/tasks/ocr.py`) was added this way and serves as a reference implementation.

---

## ECR Metric

**Ecosystem-relative Capability Ratio (ECR)** is defined as:

```
ECR = DeltaAcc / log10(L_MMLLM / L_classic)
```

Where:
- `DeltaAcc = Acc_MMLLM - Acc_classic` (accuracy difference, best MMLLM minus best classic)
- `L_MMLLM` = median inference latency of the best MMLLM (ms)
- `L_classic` = median inference latency of the best classic model (ms)

**Interpretation:**
- `ECR > 0.1`: MMLLM is *justified* — it gains enough accuracy to offset the latency cost
- `0 < ECR <= 0.1`: marginal — small accuracy gain relative to much higher latency
- `ECR <= 0`: classic preferred — classic model is faster *and* more accurate (or MMLLM gains nothing)
- `ECR = +inf`: MMLLM is faster *and* more accurate (unusual)

ECR is computed automatically by the harness and printed in the summary table.

