"""
Tests for the randomized lasso module.
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils import check_random_state

from stability_selection.randomized_lasso import RandomizedLasso, RandomizedLogisticRegression


def test_randomized_lasso():
    """Test RandomizedLasso estimator."""
    # Generate regression data
    rng = check_random_state(0)
    X = rng.normal(0, 1, (100, 10))
    beta = np.zeros(10)
    beta[:3] = 1  # First 3 features are important
    y = np.dot(X, beta) + rng.normal(0, 0.1, 100)
    
    # Test with default parameters
    estimator = RandomizedLasso(weakness=0.5, alpha=0.1)
    estimator.fit(X, y)
    
    # Check that coefficients are available
    assert hasattr(estimator, 'coef_')
    assert estimator.coef_.shape == (10,)
    
    # The important features should have larger coefficients
    top_features = np.argsort(np.abs(estimator.coef_))[-3:]
    for i in range(3):
        assert i in top_features
        
    # Test with GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            estimator_gpu = RandomizedLasso(weakness=0.5, alpha=0.1, use_gpu=True)
            estimator_gpu.fit(X, y)
            assert hasattr(estimator_gpu, 'coef_')
    except ImportError:
        pass
    
    # Test prediction
    y_pred = estimator.predict(X)
    assert y_pred.shape == (100,)


def test_randomized_logistic_regression():
    """Test RandomizedLogisticRegression estimator."""
    # Generate classification data
    X, y = make_classification(n_samples=100, n_features=20, n_informative=3, 
                             n_redundant=0, random_state=42)
    
    # Test with default parameters
    estimator = RandomizedLogisticRegression(weakness=0.5, C=1.0)
    estimator.fit(X, y)
    
    # Check that coefficients are available
    assert hasattr(estimator, 'coef_')
    assert estimator.coef_.shape == (1, 20)
    
    # Test with GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            estimator_gpu = RandomizedLogisticRegression(weakness=0.5, C=1.0, use_gpu=True)
            estimator_gpu.fit(X, y)
            assert hasattr(estimator_gpu, 'coef_')
    except ImportError:
        pass
    
    # Test prediction
    y_pred = estimator.predict(X)
    assert y_pred.shape == (100,)
    
    y_proba = estimator.predict_proba(X)
    assert y_proba.shape == (100, 2)


def test_weakness_validation():
    """Test validation of the weakness parameter."""
    # Weakness should be in (0, 1]
    
    # Test invalid values
    with pytest.raises(ValueError):
        RandomizedLasso(weakness=0).fit(np.random.random((10, 5)), np.random.random(10))
    
    with pytest.raises(ValueError):
        RandomizedLasso(weakness=1.5).fit(np.random.random((10, 5)), np.random.random(10))
    
    with pytest.raises(ValueError):
        RandomizedLasso(weakness=-0.5).fit(np.random.random((10, 5)), np.random.random(10))
    
    # Test valid value
    try:
        RandomizedLasso(weakness=1.0).fit(np.random.random((10, 5)), np.random.random(10))
        RandomizedLasso(weakness=0.5).fit(np.random.random((10, 5)), np.random.random(10))
        RandomizedLasso(weakness=0.1).fit(np.random.random((10, 5)), np.random.random(10))
    except ValueError:
        pytest.fail("RandomizedLasso raised ValueError unexpectedly for valid weakness values")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
