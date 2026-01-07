from setuptools import setup, find_packages

setup(
    name="titan_data",
    version="0.1.0",
    description="Advanced Audio Dataset Generator for Robust Deep Learning",
    long_description="A procedural audio augmentation library designed for training SOTA denoising models. Features real-time DSP synthesis, SNR control, and acoustic simulation.",
    author="breili-u",
    author_email="breili-u@email.com",
    url="https://github.com/breili-u/titan_data",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
)