"""
===========================
Randomized LASSO estimators
===========================

This module contains implementations of randomized logistic regression
and randomized LASSO regression with GPU acceleration via PyTorch.

References
----------
.. [1] Meinshausen, N. and Buhlmann, P., 2010. Stability selection.
    Journal of the Royal Statistical Society: Series B
    (Statistical Methodology), 72(4), pp.417-473.
"""
import numpy as np
import warnings

from scipy import sparse
from scipy.sparse import issparse

from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.utils import check_X_y, check_random_state

# Import PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch is not available. GPU acceleration will be disabled.")

__all__ = ['RandomizedLogisticRegression', 'RandomizedLasso']


def _rescale_data(X, weights):
    if issparse(X):
        size = weights.shape[0]
        weight_dia = sparse.dia_matrix((1 - weights, 0), (size, size))
        X_rescaled = X * weight_dia
    else:
        X_rescaled = X * (1 - weights)

    return X_rescaled


def _check_gpu_availability():
    """Check if PyTorch is available and if a CUDA-compatible GPU is available."""
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


class RandomizedLogisticRegression(LogisticRegression):
    """
    Randomized version of scikit-learns LogisticRegression class with GPU acceleration.

    Randomized LASSO is a generalization of the LASSO. The LASSO
    penalises the absolute value of the coefficients with a penalty
    term proportional to `C`, but the randomized LASSO changes the
    penalty to a randomly chosen value in the range `[C, C/weakness]`.

    Parameters
    ----------
    weakness : float
        Weakness value for randomized LASSO. Must be in (0, 1].

    use_gpu : bool, default=False
        Whether to use GPU acceleration if available.

    See also
    --------
    sklearn.linear_model.LogisticRegression : learns logistic regression
    models using the same algorithm.
    """
    def __init__(self, weakness=0.5, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1,
                 use_gpu=False):
        self.weakness = weakness
        self.use_gpu = use_gpu
        super(RandomizedLogisticRegression, self).__init__(
            penalty='l1', dual=False, tol=tol, C=C, fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling, class_weight=class_weight,
            random_state=random_state, solver=solver, max_iter=max_iter,
            multi_class=multi_class, verbose=verbose, warm_start=warm_start,
            n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        """
        if not isinstance(self.weakness, float) or not (0.0 < self.weakness <= 1.0):
            raise ValueError('weakness should be a float in (0, 1], got %s' % self.weakness)

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=[np.float64, np.float32],
                         order="C")

        n_features = X.shape[1]
        weakness = 1. - self.weakness
        random_state = check_random_state(self.random_state)

        # Generate random weights
        weights = weakness * random_state.randint(0, 1 + 1, size=(n_features,))
        
        # GPU acceleration for weight rescaling if available and requested
        if self.use_gpu and _check_gpu_availability() and not issparse(X):
            try:
                # Convert to PyTorch tensors
                X_tensor = torch.tensor(X, dtype=torch.float32, device='cuda')
                weights_tensor = torch.tensor(1 - weights, dtype=torch.float32, device='cuda')
                
                # Rescale data on GPU
                X_rescaled_tensor = X_tensor * weights_tensor
                
                # Convert back to numpy
                X_rescaled = X_rescaled_tensor.cpu().numpy()
            except Exception as e:
                warnings.warn(f"GPU acceleration failed: {str(e)}. Falling back to CPU.")
                X_rescaled = _rescale_data(X, weights)
        else:
            X_rescaled = _rescale_data(X, weights)
            
        return super(RandomizedLogisticRegression, self).fit(X_rescaled, y, sample_weight)


class RandomizedLasso(Lasso):
    """
    Randomized version of scikit-learns Lasso class with GPU acceleration.

    Randomized LASSO is a generalization of the LASSO. The LASSO penalises
    the absolute value of the coefficients with a penalty term proportional
    to `alpha`, but the randomized LASSO changes the penalty to a randomly
    chosen value in the range `[alpha, alpha/weakness]`.

    Parameters
    ----------
    weakness : float
        Weakness value for randomized LASSO. Must be in (0, 1].
        
    use_gpu : bool, default=False
        Whether to use GPU acceleration if available.

    See also
    --------
    sklearn.linear_model.Lasso : learns Lasso regression models
    using the same algorithm.
    """
    def __init__(self, weakness=0.5, alpha=1.0, fit_intercept=True, 
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic', use_gpu=False):
        self.weakness = weakness
        self.use_gpu = use_gpu
        super(RandomizedLasso, self).__init__(
            alpha=alpha, fit_intercept=fit_intercept,
            precompute=precompute, copy_X=copy_X,
            max_iter=max_iter, tol=tol, warm_start=warm_start,
            positive=positive, random_state=random_state,
            selection=selection)

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        if not isinstance(self.weakness, float) or not (0.0 < self.weakness <= 1.0):
            raise ValueError('weakness should be a float in (0, 1], got %s' % self.weakness)

        X, y = check_X_y(X, y, accept_sparse=True)

        n_features = X.shape[1]
        weakness = 1. - self.weakness
        random_state = check_random_state(self.random_state)

        # Generate random weights
        weights = weakness * random_state.randint(0, 1 + 1, size=(n_features,))
        
        # GPU acceleration for weight rescaling if available and requested
        if self.use_gpu and _check_gpu_availability() and not issparse(X):
            try:
                # Convert to PyTorch tensors
                X_tensor = torch.tensor(X, dtype=torch.float32, device='cuda')
                weights_tensor = torch.tensor(1 - weights, dtype=torch.float32, device='cuda')
                
                # Rescale data on GPU
                X_rescaled_tensor = X_tensor * weights_tensor
                
                # Convert back to numpy
                X_rescaled = X_rescaled_tensor.cpu().numpy()
            except Exception as e:
                warnings.warn(f"GPU acceleration failed: {str(e)}. Falling back to CPU.")
                X_rescaled = _rescale_data(X, weights)
        else:
            X_rescaled = _rescale_data(X, weights)
            
        return super(RandomizedLasso, self).fit(X_rescaled, y)
