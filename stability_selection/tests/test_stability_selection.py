"""
Tests for the StabilitySelection class.
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stability_selection import StabilitySelection
from stability_selection.randomized_lasso import RandomizedLasso


def test_stability_selection_classification():
    """Test StabilitySelection on a classification task."""
    # Generate data
    X, y = make_classification(n_samples=200, n_features=50, n_informative=5,
                             n_redundant=5, random_state=42)
    # Create a pipeline
    base_estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(penalty='l1', solver='liblinear'))
    ])
    # Run stability selection
    selector = StabilitySelection(
        base_estimator=base_estimator,
        lambda_name='model__C',
        lambda_grid=np.logspace(-5, -1, 10),
        n_bootstrap_iterations=20,  # Less iterations for faster testing
        n_jobs=-1,
        verbose=0
    )
    selector.fit(X, y)
    
    # Check that we have stability scores
    assert hasattr(selector, 'stability_scores_')
    assert selector.stability_scores_.shape == (X.shape[1], 10)
    
    # Check that we can get selected features
    selected = selector.get_support(indices=True)
    assert isinstance(selected, np.ndarray)
    
    # Test transform
    X_reduced = selector.transform(X)
    assert X_reduced.shape[0] == X.shape[0]
    assert X_reduced.shape[1] <= X.shape[1]


def test_stability_selection_regression():
    """Test StabilitySelection on a regression task."""
    # Generate data
    X = np.random.normal(0, 1, (100, 30))
    y = X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.1, 100)
    
    # Create a base estimator
    base_estimator = RandomizedLasso(weakness=0.5)
    
    # Run stability selection
    selector = StabilitySelection(
        base_estimator=base_estimator, 
        lambda_name='alpha',
        lambda_grid=np.linspace(0.01, 1.0, 10),
        n_bootstrap_iterations=20,  # Less iterations for faster testing
        n_jobs=-1,
        verbose=0
    )
    selector.fit(X, y)
    
    # Check that we have stability scores
    assert hasattr(selector, 'stability_scores_')
    assert selector.stability_scores_.shape == (X.shape[1], 10)
    
    # Check that top features include the true ones
    top_features = np.argsort(selector.stability_scores_.max(axis=1))[-5:]
    assert 0 in top_features or 1 in top_features


def test_input_validation():
    """Test input validation in StabilitySelection."""
    # Invalid n_bootstrap_iterations
    with pytest.raises(ValueError):
        StabilitySelection(n_bootstrap_iterations=0)
    
    # Invalid sample_fraction
    with pytest.raises(ValueError):
        StabilitySelection(sample_fraction=0)
    with pytest.raises(ValueError):
        StabilitySelection(sample_fraction=1.5)
    
    # Invalid threshold
    with pytest.raises(ValueError):
        StabilitySelection(threshold=0)
    with pytest.raises(ValueError):
        StabilitySelection(threshold=1.5)
    
    # Invalid bootstrap_func
    with pytest.raises(ValueError):
        StabilitySelection(bootstrap_func='invalid_name')


def test_get_support():
    """Test get_support method."""
    # Generate data
    X = np.random.normal(0, 1, (100, 10))
    y = X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.1, 100)
    
    # Create mock stability scores
    selector = StabilitySelection()
    selector.stability_scores_ = np.zeros((10, 5))
    selector.stability_scores_[0, :] = 0.9  # Feature 0 always selected
    selector.stability_scores_[1, 0] = 0.8  # Feature 1 selected sometimes
    
    # Test default threshold
    support = selector.get_support()
    assert support[0]  # Feature 0 should be selected
    assert support[1]  # Feature 1 should be selected
    assert not support[2:]  # Other features should not be selected
    
    # Test custom threshold
    support = selector.get_support(threshold=0.85)
    assert support[0]  # Feature 0 should be selected
    assert not support[1]  # Feature 1 should not be selected
    assert not support[2:]  # Other features should not be selected
    
    # Test indices
    indices = selector.get_support(indices=True)
    assert 0 in indices
    assert 1 in indices
    assert len(indices) == 2


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
