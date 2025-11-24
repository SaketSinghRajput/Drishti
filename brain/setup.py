from setuptools import find_packages, setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name="accident-prediction-system",
    version="0.1.0",
    description="Multimodal accident prediction system for edge deployment",
    author="Project Team",
    author_email="maintainer@example.com",
    url="https://example.com/accident-prediction",
    license="MIT",
    packages=find_packages(exclude=("tests", "docs")),
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "edge": ["onnxruntime"],
        "training": ["tensorflow", "torch"],
        "llm": ["google-generativeai"],
        # Battery monitoring system dependencies
        "battery": ["python-can", "flask", "sqlalchemy", "RPi.GPIO", "gpiozero"],
    },
    # entry_points left intentionally empty until CLI modules are implemented.
    entry_points={},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
