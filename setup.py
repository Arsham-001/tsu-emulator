"""
NOTE: Primary build configuration has moved to pyproject.toml.
setup.py retained for backward compatibility and tooling compatibility.
Do not remove existing functionality.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tsu-platform",
    version="0.1.0",
    author="Arsham Rocky",
    author_email="arsham.rocky21@gmail.com",
    description="Thermodynamic Sampling Unit emulator for probabilistic computing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Arsham-001/tsu-emulator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.3.0",
        "scipy>=1.16.0",
        "matplotlib>=3.10.0",
    ],
    extras_require={
        "viz": [
            "plotly>=6.4.0",
        ],
        "dev": [
            "pytest>=9.0.0",
            "pytest-timeout>=2.4.0",
        ],
    },
)
