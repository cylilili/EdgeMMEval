## 5. `HD` (Hazard Detection)
1. Create virtual environment using `venv`. Run:
    1. `python3 -m venv env-hd`
    2. `source env-hd/bin/activate`
2. Install dependencies. Assuming `env-hd` is active, run: `pip install -r requirements.txt`
3. Download dataset. To run the experiment on the DetectiumFire videos, you have to download the dataset (https://www.kaggle.com/datasets/yimengfuyao/detectiumfire). For convenience, you can simply download the dataset by running the following command: `python download_detectiumfire_dataset.py`
4. Run experiment with: `python 1_hazard_detection_detectium_fire.py`
5. To directly check the results, run the visualization script using: `2_visualization_hazard_detection_detectium_fire.py`