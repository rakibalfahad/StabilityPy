# Compatibility Notes for StabilityPy

## Scikit-learn Compatibility Updates

### Removed Deprecated Imports
- Removed import of deprecated `_preprocess_data` from `sklearn.linear_model.base` in `randomized_lasso.py`. This function was removed in scikit-learn 1.3.0 and is no longer available.

### Parameter Updates
- Removed the deprecated `normalize` parameter from `RandomizedLasso` class in `randomized_lasso.py`. This parameter was deprecated in scikit-learn 1.0 and removed in scikit-learn 1.3.0.
- The `RandomizedLasso` class has been updated to work with scikit-learn 1.3.0 and later versions.

### Test Fixes
- Fixed `test_get_support` to properly handle array comparisons using `np.any(array)` instead of direct boolean evaluation.
- Updated the validation in `StabilitySelection` class to perform parameter validation during initialization, ensuring that parameters like `n_bootstrap_iterations`, `sample_fraction`, and `threshold` are properly validated.
- Added validation for `bootstrap_func` parameter during initialization to ensure proper validation when a string is provided.

## GPU Acceleration
- The library now supports GPU acceleration for stability selection computations when PyTorch is available.
- A fallback mechanism has been implemented to use CPU when GPU is requested but not available.

## Known Issues
- Some flake8 linting issues remain in the codebase, mainly related to whitespace and line length. These do not affect functionality but could be addressed in future updates.

## Requirements
The library now requires:
- scikit-learn >= 1.3.0
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- torch >= 1.10.0 (optional, for GPU acceleration)
- joblib >= 1.0.0
- tqdm >= 4.60.0
