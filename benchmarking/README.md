## 1. `OC` (Object Counting) reproduce experiments
1. Create virtual environment using `venv` and navigate to `1_object_counting`. Run:
    1. `python3 -m venv env-oc`
    2. `source env-oc/bin/activate`
2. Install dependencies. Assuming `env-oc` is active, run:
    1. `pip install -r requirements.txt`
3. Run script using `python 1_object_counting_coco.py`

## 2. `ASR` (Automatic Speech Recognition)
1. Create virtual environment using `venv` and navigate to `2_automatic_speech_recognition`. Run:
    1. `python3 -m venv env-asr`
    2. `source env-asr/bin/activate`
2. Install dependencies. Assuming `env-asr` is active, run:
    1. `pip install -r requirements.txt`
3. Run ASR after you `cd src/` with:
```
python 1_automatic_speech_recognition_librispeech.py \
  --split test.clean --num_samples 2620 --seed 0 \
  --whisper_models tiny,base,small,medium,large-v3,large-v3-turbo \
  --use_gemma_e2b --use_gemma_e4b \
  --use_vosk --vosk_models small-en-us-0.15,en-us-0.22 \
  --latency_scale log \
  --log_path results_asr_log_2620.csv --save_jsonl
```
4. For visualizations (with the existing results from the paper `results_asr_log_2620.csv`):
```
python 2_visualization_automatic_speech_recognition_librispeech.py --csv results_asr_log_2620.csv --out_dir ./figs --latency_scale log
  ```

### Data download
The LibriSpeech data is automatically downloaded as you run the script (it amounts to roughly 60GiB).

## 3. `IC` (Image Classification)
1. Create virtual environment using `venv` and navigate to `3_image_classification`. Run:
    1. `python3 -m venv env-ic`
    2. `source env-ic/bin/activate`
2. Install dependencies. Assuming `env-ic` is active, run:
    1. `pip install -r requirements.txt`
3. Run IC after you `cd src/` with:
```
python 1_image_classification_imagenette.py \
  --dataset frgfm/imagewoof --config 160px --split validation --num_samples 3925 --seed 0 \
  --tv_models mobilenet_v3_large,resnet50 \
  --use_qwen3_vl_2b --use_gemma_e4b --use_gemma_e2b --use_ministral3_3b
```
4. For visualizations (with the existing results from the paper `results_imgcls_log_3925_validation.csv`):
```
python 2_visualizations_image_classification_imagenette.py \
  --log_csv results_imgcls_log_3925_validation.csv \
  ```

## 4. `AR` (Action Recognition)
1. Create virtual environment using `venv` and navigate to `4_action_recognition`. Run:
    1. `python3 -m venv env-ar`
    2. `source env-ar/bin/activate`
2. Install dependencies. Assuming `env-ar` is active, run: `pip install -r requirements.txt`
3. Run AR after you `cd src/` with:
```
python 1_action_recognition_kinetics.py \
  --dataset nateraw/kinetics-mini --split validation --num_samples 50 --seed 0 \
  --tv_models r3d_18,mc3_18,r2plus1d_18 \
  --use_gemma_e2b --use_gemma_e4b \
  --device auto \
  --clip_len 16 --stride 4 \
  --gemma_frames 8 --grid_cols 4 \
  --latency_scale log \
  --latency_plot_path latency_ar_kinetics.png --acc_plot_path accuracy_ar_kinetics.png \
  --log_path ar_kinetics_log.csv --save_jsonl
```
4. For visualizations (with the existing results from the paper `ar_kinetics_log.csv`):
```
python 2_visualizations_action_recognition_kinetics.py --log_csv ar_kinetics_log.csv --latency_plot latency_ar_kinetics.png --acc_plot accuracy_ar_kinetics.png --latency_scale log
```

## 5. `HD` (Hazard Detection)
1. Create virtual environment using `venv` and navigate to `5_hazard_detection`. Run:
    1. `python3 -m venv env-hd`
    2. `source env-hd/bin/activate`
2. Install dependencies. Assuming `env-hd` is active, run: `pip install -r requirements.txt`
3. Download dataset. To run the experiment on the DetectiumFire videos, you have to download the dataset (https://www.kaggle.com/datasets/yimengfuyao/detectiumfire). For convenience, you can simply download the dataset by running the following command: `python download_detectiumfire_dataset.py`
4. Run experiment with: `python 1_hazard_detection_detectium_fire.py`
5. To directly check the results, run the visualization script using: `2_visualization_hazard_detection_detectium_fire.py`