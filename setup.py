from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neural-style-transfer-suite",
    version="1.0.0",
    author="Neural Style Transfer Suite",
    description="Comprehensive neural style transfer suite for Windows 11",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/coco11211/ai-medical-diagnosis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows :: Windows 11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "click>=8.1.0",
        "tqdm>=4.66.0",
        "PyYAML>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "nst=nst_suite.cli:main",
            "nst-train=nst_suite.training.train:main",
            "nst-video=nst_suite.video.process:main",
        ],
    },
)
