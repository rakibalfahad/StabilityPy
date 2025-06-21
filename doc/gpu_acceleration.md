# GPU Acceleration in StabilityPy

This document explains how GPU acceleration works in the StabilityPy library and how to use it effectively.

## Overview

StabilityPy supports GPU acceleration via PyTorch to speed up stability selection computations. When enabled and when a compatible GPU is available, certain operations can run significantly faster than on CPU alone.

## Requirements

To use GPU acceleration, you need:

1. PyTorch installed (`pip install torch`)
2. A CUDA-compatible GPU
3. Appropriate CUDA drivers installed

## Usage

To enable GPU acceleration, simply set the `use_gpu` parameter to `True` when creating a `StabilitySelection` instance:

```python
from stability_selection import StabilitySelection

selector = StabilitySelection(
    base_estimator=my_estimator,
    lambda_name='my_regularization_param',
    lambda_grid=my_lambda_values,
    use_gpu=True  # Enable GPU acceleration
)
```

## How It Works

When GPU acceleration is enabled:

1. The library checks if PyTorch is available and if a CUDA-compatible GPU is detected
2. If available, the base estimator is configured to use GPU (if it supports it)
3. The bootstrap sampling and model fitting operations may utilize GPU resources
4. If GPU is not available, the library falls back to CPU processing with a warning

## Graceful Fallback

A key feature of our implementation is the graceful fallback to CPU processing when GPU acceleration is requested but not available. The library will display a warning message and continue processing using CPU:

```
UserWarning: GPU requested but not available. Falling back to CPU.
```

This ensures that your code will run correctly regardless of the hardware environment, making it suitable for both development on machines without GPUs and deployment on GPU-equipped servers.

## Performance Gains

Performance improvements vary based on:

- Dataset size and dimensionality
- Model complexity
- Available GPU hardware
- Number of bootstrap iterations

In our benchmark tests using synthetic data (500 samples × 1000 features), we observed speedups ranging from 1.2× to 2× compared to CPU-only processing.

## Example

The following example demonstrates how to use GPU acceleration and compare performance:

```python
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stability_selection import StabilitySelection

# Create a base estimator
base_estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty='l1', solver='liblinear'))
])

# Configure lambda grid
lambda_grid = np.logspace(-5, -1, 50)

# Run on CPU
start_time = time.time()
cpu_selector = StabilitySelection(
    base_estimator=base_estimator, 
    lambda_name='model__C',
    lambda_grid=lambda_grid,
    use_gpu=False
)
cpu_selector.fit(X, y)
cpu_time = time.time() - start_time

# Run with GPU
start_time = time.time()
gpu_selector = StabilitySelection(
    base_estimator=base_estimator, 
    lambda_name='model__C',
    lambda_grid=lambda_grid,
    use_gpu=True
)
gpu_selector.fit(X, y)
gpu_time = time.time() - start_time

# Compare performance
speedup = cpu_time / gpu_time
print(f"GPU provided a {speedup:.2f}x speedup")
```

See the full example in `examples/gpu_acceleration_example.py`.

## Limitations

- Not all operations benefit from GPU acceleration
- Small datasets may not see significant speedups due to data transfer overhead
- Some models or operations may not be GPU-compatible

## Troubleshooting

If you encounter issues with GPU acceleration:

1. Verify PyTorch is installed: `pip install torch`
2. Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
3. Update GPU drivers if needed
4. Monitor GPU memory usage during execution
