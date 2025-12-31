#!/usr/bin/env python
"""
Setup script for MLOps Energy Demand Forecasting project.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="mlops-energy-demand-forecasting",
    version="1.0.0",
    author="MLOps Team",
    author_email="mlops@example.com",
    description="MLOps pipeline for energy demand forecasting with monitoring and deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/mlops-energy-demand-forecasting",

    packages=find_packages(where="src"),
    package_dir={"": "src"},

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    python_requires=">=3.8",

    install_requires=read_requirements("requirements-fixed.txt"),

    extras_require={
        "dev": [
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.7.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },

    entry_points={
        "console_scripts": [
            "energy-forecast-api=deployment.fastapi_app:main",
            "energy-forecast-train=scripts.training.train_script:main",
            "energy-forecast-monitor=scripts.monitoring.run_monitoring:main",
        ],
    },

    include_package_data=True,

    zip_safe=False,
)
