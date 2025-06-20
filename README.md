# StabilityPy

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

A modern implementation of stability selection for feature selection, with support for GPU acceleration via PyTorch and parallel processing for CPU.

## Introduction

Stability selection is a method for feature selection introduced by Meinshausen and Bühlmann (2010). The core idea is to apply a feature selection algorithm repeatedly to random subsamples of the data and select only those features that are consistently selected across many subsamples.

### Theory

Stability selection works as follows:

1. Create multiple subsamples of your data by randomly selecting a subset of observations.
2. For each subsample, run a feature selection algorithm (often a penalized regression like LASSO) across a range of regularization parameters.
3. For each feature, calculate its selection probability (stability score) as the fraction of subsamples where it was selected.
4. Choose features with stability scores above a user-defined threshold.

This approach has several advantages:
- It provides control over the family-wise error rate of including false positives
- It is more robust to small changes in the data
- It works with many different base feature selection methods
- It reduces the sensitivity to the choice of regularization parameter

In the randomized LASSO variant, the penalty term for each feature is randomly scaled, which adds another layer of robustness.

## Installation

### From PyPI (Recommended)

```bash
pip install stability-selection
```

### From Source

To install the latest development version:

```bash
git clone https://github.com/yourusername/StabilityPy.git
cd StabilityPy
pip install -e .
```

For development, install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Features

- **GPU Acceleration**: Uses PyTorch for GPU-accelerated computations when available
- **Parallel Processing**: Efficient multi-core CPU utilization for bootstrap iterations
- **Multiple Bootstrapping Strategies**:
  - Subsampling without replacement (default)
  - Complementary pairs subsampling
  - Stratified bootstrapping for imbalanced classification
- **Scikit-learn Compatible**: Works with scikit-learn pipelines and cross-validation

## Example Usage

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from stability_selection import StabilitySelection

# Create a pipeline with a base estimator
base_estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty='l1', solver='liblinear'))
])

# Initialize stability selection with the base estimator
selector = StabilitySelection(
    base_estimator=base_estimator,
    lambda_name='model__C',
    lambda_grid=np.logspace(-5, -1, 50),
    n_jobs=-1,  # Use all available CPU cores
    use_gpu=True  # Use GPU if available
)

# Fit the selector to your data
selector.fit(X, y)
```

### Advanced Example with Visualizations

The package includes comprehensive example scripts that demonstrate various use cases:

```bash
# Basic stability selection example
python examples/stability_selection_example.py

# Example with GPU acceleration
python examples/gpu_acceleration_example.py

# Example with synthetic data visualization
python examples/synthetic_data_visualization.py
```

The stability selection example will generate a visualization of stability paths:

![Stability Paths Example](stability_path.png)

## GPU Acceleration

StabilityPy provides GPU acceleration via PyTorch for faster computation, especially useful for large datasets:

```python
from stability_selection import StabilitySelection, RandomizedLasso

# For regression tasks, use RandomizedLasso with GPU acceleration
estimator = RandomizedLasso(weakness=0.5, use_gpu=True)
selector = StabilitySelection(
    base_estimator=estimator,
    lambda_name='alpha',
    lambda_grid=np.linspace(0.001, 0.5, num=100),
    threshold=0.9,
    use_gpu=True,  # Enable GPU acceleration
    n_jobs=-1  # Use all CPU cores for operations that can't be GPU accelerated
)
selector.fit(X, y)
```

## Development

The project follows standard Python development practices with tools for code quality and testing.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/StabilityPy.git
cd StabilityPy

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality Tools

We use several tools to maintain code quality:

```bash
# Format code
black stability_selection examples --line-length=100
isort stability_selection examples

# Lint code
flake8 stability_selection examples --max-line-length=100

# Run tests
pytest stability_selection/tests

# Run tests with coverage
pytest --cov=stability_selection --cov-report=term
```

For more details, see the [Development Guide](DEVELOPMENT.md) and [Standardization Summary](STANDARDIZATION.md).

## References

[1] Meinshausen, N. and Bühlmann, P. (2010). Stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(4), pp.417-473. [Link to paper](https://arxiv.org/pdf/0809.2932)

[2] Shah, R.D. and Samworth, R.J. (2013). Variable selection with error control: another look at stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 75(1), pp.55-80.

## Requirements

- Python 3.8+
- NumPy >= 1.24.0
- SciPy >= 1.11.0
- scikit-learn >= 1.3.0
- PyTorch >= 2.0.0 (optional, for GPU acceleration)
- joblib >= 1.3.0
- tqdm >= 4.65.0
- matplotlib >= 3.7.0 (for visualization)
- seaborn >= 0.12.0 (for visualization)

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is a modernized version of the original stability-selection package by Thomas Huijskens, with added features for GPU acceleration, improved parallel processing, and standardized code quality.