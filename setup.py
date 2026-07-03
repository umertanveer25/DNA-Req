from setuptools import setup, find_packages

setup(
    name="dna-nfr-classification",
    version="1.0.0",
    author="Umer Tanveer",
    author_email="talktoumer94@gmail.com",
    description="DNA-Inspired Feature Engineering for Software Requirements Classification",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/talktoumer94/DNA-Inspired-NFR-Classifier",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "sentence-transformers>=2.2.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
)
