from setuptools import setup, find_packages

with open("readme.qmd", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="py_distcomp",
    version="0.1.0",
    author="Chris Russell",
    author_email="your.email@leeds.ac.uk",
    description="A professional Python library for comprehensive statistical distribution comparison and visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chris-r-uol/py_distcomp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "py-distcomp-demo=app:main",
        ],
    },
    keywords="statistics, distribution, fitting, visualization, qq-plot, probability",
    project_urls={
        "Bug Reports": "https://github.com/chris-r-uol/py_distcomp/issues",
        "Source": "https://github.com/chris-r-uol/py_distcomp",
        "Documentation": "https://github.com/chris-r-uol/py_distcomp#readme",
    },
)
