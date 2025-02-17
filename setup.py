from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pricebin",
    version="0.0",
    description="A comprehensive Python package for stock price tracking and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jerald Achaibar",
    url="https://github.com/jach8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.19.0",
        "yfinance>=0.1.63",
        "matplotlib>=3.3.0",
        "pandas_datareader>=0.9.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pricebin-init=pricebin.Initialize:Initialize",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    # Include additional files
    package_data={
        "pricebin": ["connections.json"],
    },
    # Add dependencies required for running tests
    tests_require=[
        "pytest>=6.0",
        "pytest-cov>=2.0",
    ],
    # Enable running tests via setup.py test
    test_suite="tests",
    # Project metadata
    project_urls={
        "Bug Reports": "https://github.com/jach8/PriceBin/issues",
        "Documentation": "https://github.com/jach8/PriceBin#readme",
        "Source Code": "https://github.com/jach8/PriceBin",
    },
    # Keywords for PyPI
    keywords=[
        "finance",
        "stock market",
        "technical analysis",
        "trading",
        "investment",
        "data analysis",
        "sqlite",
    ],
)