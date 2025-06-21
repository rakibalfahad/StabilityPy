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

For GPU acceleration, install with GPU dependencies:

```bash
pip install -e ".[gpu]"
```

## Features

- **GPU Acceleration**: Uses PyTorch for GPU-accelerated computations when available
- **Parallel Processing**: Efficient multi-core CPU utilization for bootstrap iterations
- **Multiple Bootstrapping Strategies**:
  - Subsampling without replacement (default)
  - Complementary pairs subsampling
  - Stratified bootstrapping for imbalanced classification
- **Scikit-learn Compatible**: Works with scikit-learn pipelines and cross-validation
- **CSV/CSV.GZ Processing**: Direct support for tabular data formats
- **Automated Feature Selection**: Process tabular data and visualize results with a single command
- **Model Fine-tuning**: Automatically fine-tune models with selected features and compare to baselines
- **Synthetic Data Generation**: Generate controlled datasets for testing and benchmarking

## Documentation

- [API Reference](doc/api_reference.md)
- [GPU Acceleration](doc/gpu_acceleration.md)
- [Theory and Implementation](blog/stability_selection_theory.md)

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

# Get selected features
selected_features = selector.get_support(indices=True)
```

### Data Processing and Analysis

StabilityPy now includes a comprehensive script for processing tabular data and running the full stability selection workflow:

```bash
# For classification problems
python stability_processor.py --input data.csv --output results_dir --problem_type classification

# For regression problems
python stability_processor.py --input data.csv.gz --output results_dir --problem_type regression --use_gpu
```

The script will:
1. Load and preprocess your data
2. Run stability selection to identify important features
3. Fine-tune a model using only the selected features
4. Compare performance with a baseline model using all features
5. Generate visualizations and save all results to the output directory

### Synthetic Data Generation

You can generate synthetic datasets with controlled properties using the included generator:

```bash
# Generate a classification dataset
python synthetic_data_generator.py --output data.csv --problem_type classification --n_samples 1000 --n_features 100 --n_informative 10

# Generate a compressed regression dataset
python synthetic_data_generator.py --output data.csv.gz --problem_type regression --n_samples 2000 --n_features 500 --n_informative 20 --compress
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

### Complete Workflow Example

Here's a complete workflow from data generation to model fine-tuning:

```bash
# 1. Generate synthetic data
python synthetic_data_generator.py \
    --output data/synthetic_classification.csv \
    --problem_type classification \
    --n_samples 1000 \
    --n_features 100 \
    --n_informative 10 \
    --noise 0.1

# 2. Run stability selection with fine-tuning
python stability_processor.py \
    --input data/synthetic_classification.csv \
    --output results/synthetic_classification \
    --problem_type classification \
    --n_bootstrap 100 \
    --use_gpu
```

## Output Files and Visualizations

The `stability_processor.py` script produces a comprehensive set of outputs:

- **selected_features.csv**: CSV file with selected features and their stability scores
- **feature_importance.csv**: Feature importances from the fine-tuned model
- **performance_metrics.csv**: Performance comparison between selected features and all features
- **stability_selection_results.pkl**: Pickled results object with all stability scores
- **fine_tuned_model.pkl**: Trained model using only selected features
- **baseline_model.pkl**: Trained model using all features

Visualizations:
- **stability_paths.png**: Plot of stability scores across regularization parameters
- **stability_heatmap.png**: Heatmap of stability scores for top features
- **performance_comparison.png**: Bar chart comparing model performance
- **feature_importance.png**: Bar chart of feature importances from fine-tuned model

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
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- scikit-learn >= 1.0.0
- PyTorch >= 1.10.0 (optional, for GPU acceleration)
- joblib >= 1.0.0
- tqdm >= 4.60.0
- matplotlib >= 3.3.0 (for visualization)
- seaborn >= 0.11.0 (for visualization)

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is a modernized version of the original stability-selection package by Thomas Huijskens, with added features for GPU acceleration, improved parallel processing, and standardized code quality.