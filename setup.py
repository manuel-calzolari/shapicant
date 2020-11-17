from io import open
from os import path

from setuptools import find_packages, setup

import shapicant

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="shapicant",
    version=shapicant.__version__,
    description="Feature selection package based on SHAP and target permutation, for pandas and Spark",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/manuel-calzolari/shapicant",
    download_url="https://github.com/manuel-calzolari/shapicant/releases",
    author="Manuel Calzolari",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=["shap>=0.36.0", "numpy", "pandas", "scikit-learn", "tqdm"],
    extras_require={
        "spark": ["pyspark>=2.4", "pyarrow"],
    },
)
