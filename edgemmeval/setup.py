from setuptools import setup, find_packages

setup(
    name="edgemmeval",
    version="1.0.0",
    description="EdgeMMEval: Benchmark harness for evaluating MMLLMs vs. specialized edge models",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "ultralytics>=8.0.0",
        "openai-whisper>=20231117",
        "vosk>=0.3.45",
        "soundfile>=0.12.1",
        "ollama>=0.2.0",
        "transformers>=4.37.0",
        "accelerate>=0.26.0",
        "psutil>=5.9.0",
        "opencv-python>=4.8.0",
        "pandas>=1.5.0",
        "matplotlib>=3.7.0",
    ],
)
