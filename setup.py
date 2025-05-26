"""
Setup script for GridSource Bank Energy Banking Pipeline
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gridsource",
    version="1.0.0",
    author="GridSource Bank Data Team",
    description="End-to-end data pipeline for energy banking liquidity forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Read from requirements.txt
        line.strip() for line in open("requirements.txt") 
        if line.strip() and not line.startswith("#")
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-mock>=3.10.0",
            "pytest-cov>=4.0.0",
            "requests-mock>=1.10.0",
            "moto>=4.1.0",
            "apache-airflow[amazon,snowflake]>=2.5.0",
        ],
    },
)