# Development Guide

This document provides instructions for developers who want to contribute to the StabilityPy project.

## Setting Up Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/StabilityPy.git
   cd StabilityPy
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Code Style

This project follows PEP 8 guidelines with some modifications:
- Line length: 100 characters
- Code is formatted using Black and isort

To automatically format code:
```bash
black stability_selection examples --line-length=100
isort stability_selection examples
```

### Running Tests

```bash
pytest stability_selection/tests
```

To run tests with coverage:
```bash
pytest --cov=stability_selection --cov-report=term --cov-report=html
```

## GPU Support

The project supports GPU acceleration via PyTorch. To enable GPU acceleration:

1. Make sure you have a CUDA-compatible GPU
2. Install PyTorch with CUDA support
3. Use the `use_gpu=True` parameter when initializing StabilitySelection

## Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality. The hooks perform the following checks:
- Trailing whitespace
- File ending in newline
- YAML/JSON syntax check
- No large files added
- No merge conflicts
- No debug statements
- Python syntax check
- Flake8 (PEP 8 compliance)
- isort (import sorting)
- Black (code formatting)
- pyupgrade (modernize Python code)

## Documentation

Please document your code following the NumPy docstring style. Example:

```python
def my_function(param1, param2):
    """
    Short description of the function.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2

    Returns
    -------
    type
        Description of return value
    """
    return result
```
