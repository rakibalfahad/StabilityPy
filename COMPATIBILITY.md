# Compatibility Fixes

This document tracks compatibility fixes made to ensure the codebase works with newer versions of dependencies.

## 1. Removed Deprecated Import

**Issue**: The code was importing `_preprocess_data` from `sklearn.linear_model.base`, but this function has been moved or removed in newer versions of scikit-learn.

**Fix**: Removed the unused import from `stability_selection/randomized_lasso.py`. The function wasn't actually used in the code, so removing it had no impact on functionality.

**File changed**: 
- `/home/ralfahad/StabilityPy/stability_selection/randomized_lasso.py`

## 2. Updated Dependency Requirements

**Issue**: The initial version requirements were too strict, requiring newer versions of numpy, scipy, and other libraries that might not be available for all Python environments.

**Fix**: Lowered the version requirements to be more compatible with a wider range of Python environments:
- numpy: 1.24.0 → 1.20.0
- scipy: 1.11.0 → 1.7.0
- scikit-learn: 1.3.0 → 1.0.0
- Other dependencies were also updated to lower versions

**Files changed**:
- `/home/ralfahad/StabilityPy/requirements.txt`
- `/home/ralfahad/StabilityPy/setup.py`
- `/home/ralfahad/StabilityPy/README.md`

## 3. Fixed Setup Process

**Issue**: The setup.py file was trying to import numpy and scipy during installation, but didn't ensure they were installed first.

**Fix**: 
- Removed the try/except blocks that were causing errors
- Added `setup_requires` parameter to ensure numpy and scipy are installed before setup
- Added checks to make sure numpy and scipy are in the install_requires list

**File changed**:
- `/home/ralfahad/StabilityPy/setup.py`

## 4. Removed Deprecated `normalize` Parameter

**Issue**: The `normalize` parameter has been deprecated and removed from scikit-learn's Lasso class in newer versions.

**Fix**: Removed the `normalize` parameter from the RandomizedLasso class initialization and super() call.

**File changed**:
- `/home/ralfahad/StabilityPy/stability_selection/randomized_lasso.py`

## 5. Added GPU Acceleration with Graceful Fallback

**Feature**: Added GPU acceleration via PyTorch for improved performance on compatible hardware.

**Implementation**:
- Added PyTorch integration for GPU-accelerated computations
- Implemented automatic detection of CUDA-compatible GPUs
- Created a graceful fallback mechanism to CPU processing when GPU is not available
- Added the `use_gpu` parameter to `StabilitySelection` class

**Compatibility Considerations**:
- The library works correctly in all environments, with or without a GPU
- When PyTorch is not installed, a warning is displayed and GPU acceleration is disabled
- When a GPU is requested but not available, a warning is displayed and processing falls back to CPU

**Files changed**:
- `/home/ralfahad/StabilityPy/stability_selection/stability_selection.py`
- `/home/ralfahad/StabilityPy/stability_selection/__init__.py`
- `/home/ralfahad/StabilityPy/requirements.txt` (added PyTorch as an optional dependency)

**Documentation**:
- Created `/home/ralfahad/StabilityPy/doc/gpu_acceleration.md` with detailed information
- Updated README.md to mention GPU acceleration capability
- Enhanced blog post with GPU acceleration details

**Examples**:
- Added `/home/ralfahad/StabilityPy/examples/gpu_acceleration_example.py` demonstrating GPU vs CPU performance
- Created `/home/ralfahad/StabilityPy/examples/gpu_benchmark_comparison.py` for detailed performance analysis
