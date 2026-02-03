"""Setup script for Z-Image World Model."""

from setuptools import setup, find_packages

setup(
    name="z-image-world",
    version="0.1.0",
    description="Interactive world model using Z-Image and StreamDiffusion",
    author="Z-Image World Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.6.0",
        "torchvision>=0.21.0",
        "accelerate>=1.0.0",
        "transformers>=4.45.0",
        "safetensors>=0.4.0",
        "einops>=0.8.0",
        "omegaconf>=2.3.0",
        "pyyaml>=6.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "training": [
            "peft>=0.13.0",
            "wandb>=0.18.0",
        ],
        "demo": [
            "pygame>=2.6.0",
            "gradio>=5.0.0",
        ],
        "video": [
            "opencv-python>=4.10.0",
            "imageio>=2.36.0",
            "imageio-ffmpeg>=0.5.0",
        ],
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.24.0",
        ],
        "all": [
            "peft>=0.13.0",
            "wandb>=0.18.0",
            "pygame>=2.6.0",
            "gradio>=5.0.0",
            "opencv-python>=4.10.0",
            "imageio>=2.36.0",
            "imageio-ffmpeg>=0.5.0",
            "pytest>=8.0.0",
            "pytest-asyncio>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "z-image-world-demo=demo.interactive_app:main",
        ],
    },
)
