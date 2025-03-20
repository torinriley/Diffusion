from setuptools import setup, find_packages

setup(
    name="custom-diffusion",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "gradio>=4.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "huggingface_hub>=0.19.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
    ],
) 