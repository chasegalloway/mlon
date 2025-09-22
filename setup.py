from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlon",
    version="1.2.0",
    description="A comprehensive utility package for machine learning development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chase Galloway",
    author_email="chase.h.galloway21@gmail.com",
    url="https://github.com/chasegalloway/mlon", 
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.1.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'joblib>=1.0.0',
        'scipy>=1.6.0',
        'click>=8.0.0',
        'reportlab>=3.6.0'
    ],
    entry_points={
        'console_scripts': [
            'mlon=mlon.cli:cli',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    project_urls={
        "Bug Tracker": "https://github.com/chasegalloway/mlutils/issues",
        "Documentation": "https://github.com/chasegalloway/mlutils#readme",
        "Source Code": "https://github.com/chasegalloway/mlutils",
    },
)
