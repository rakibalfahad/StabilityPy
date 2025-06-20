"""
===============================
Bootstrap helper functions
===============================

This module contains helper functions for stability_selection.py
 that do bootstrap sampling
"""

import numpy as np
from joblib import Parallel, delayed

from sklearn.utils.random import sample_without_replacement
from sklearn.utils.multiclass import type_of_target


__all__ = [
    'bootstrap_without_replacement',
    'complementary_pairs_bootstrap',
    'stratified_bootstrap'
]


def bootstrap_without_replacement(y, n_subsamples, random_state=None):
    """
    Bootstrap without replacement, irrespective of label. It is a wrapper around
    sklearn.utils.random.sample_without_replacement.

    Parameters
    ----------
    y : array of size [n_subsamples,]
        True labels
    n_subsamples : int
        Number of subsamples in the bootstrap sample
    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    out : array of size [n_subsamples,]
            The sampled subsets of integer. The subset of selected integer might
            not be randomized, see the method argument.
    """
    n_samples = y.shape[0]
    return sample_without_replacement(n_samples, n_subsamples,
                                      random_state=random_state)


def complementary_pairs_bootstrap(y, n_subsamples, random_state=None):
    """
    Complementary pairs bootstrap. Two subsamples A and B are generated, such
    that |A| = n_subsamples, the union of A and B equals {0, ..., n_samples - 1},
    and the intersection of A and B is the empty set. Samples irrespective of
    label.

    Parameters
    ----------
    y : array of size [n_subsamples,]
        True labels
    n_subsamples : int
        Number of subsamples in the bootstrap sample
    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    A : array of size [n_subsamples,]
            The sampled subsets of integer. The subset of selected integer
            might not be randomized, see the method argument.
    B : array of size [n_samples - n_subsamples,]
            The complement of A.
    """
    n_samples = y.shape[0]
    subsample = bootstrap_without_replacement(y, n_subsamples, random_state)
    complementary_subsample = np.setdiff1d(np.arange(n_samples), subsample)

    return subsample, complementary_subsample


def stratified_bootstrap(y, n_subsamples, random_state=None):
    """
    Bootstrap without replacement, performed separately for each group in y.

    Parameters
    ----------
    y : array of size [n_subsamples,]
        True labels
    n_subsamples : int
        Number of subsamples in the bootstrap sample
    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    out : array of size [n_subsamples,]
            The sampled subsets of integer. The subset of selected integer might
            not be randomized, see the method argument.
    """
    type_of_target_y = type_of_target(y)
    allowed_target_types = ('binary', 'multiclass')
    if type_of_target_y not in allowed_target_types:
        raise ValueError(
            'Supported target types are: {}. Got {!r} instead.'.format(
                allowed_target_types, type_of_target_y))

    unique_y, y_counts = np.unique(y, return_counts=True)
    y_counts_relative = y_counts / y_counts.sum()
    y_n_samples = np.int32(np.round(y_counts_relative * n_subsamples))

    # the above should return grouped subsamples which approximately sum up
    # to n_subsamples but may not work out exactly due to rounding errors.
    # If this is the case, adjust the count of the largest class
    if y_n_samples.sum() != n_subsamples:
        delta = n_subsamples - y_n_samples.sum()
        majority_class = np.argmax(y_counts)
        y_n_samples[majority_class] += delta

    all_selected = np.array([], dtype=np.int32)
    for i, u in enumerate(unique_y):
        indices = np.where(y == u)[0]
        selected_indices = indices[bootstrap_without_replacement(indices,
                                                                 y_n_samples[i],
                                                                 random_state)]
        all_selected = np.concatenate((all_selected, selected_indices))

    return all_selected


def parallel_bootstrap_samples(n_bootstrap_iterations, bootstrap_func, y, n_subsamples, random_state=None, n_jobs=1):
    """
    Generate bootstrap samples in parallel.
    
    Parameters
    ----------
    n_bootstrap_iterations : int
        Number of bootstrap iterations to generate
    bootstrap_func : callable
        Function to generate bootstrap samples
    y : array-like
        Target variable to stratify if needed
    n_subsamples : int
        Number of samples in each bootstrap
    random_state : int, RandomState instance or None
        Random state for reproducibility
    n_jobs : int
        Number of parallel jobs to run
        
    Returns
    -------
    list of bootstrap samples
    """
    if n_jobs == 1:
        samples = []
        for i in range(n_bootstrap_iterations):
            sample = bootstrap_func(y, n_subsamples, random_state)
            if isinstance(sample, tuple):
                samples.extend(sample)
            else:
                samples.append(sample)
        return samples
    
    # For parallel processing, we need to ensure different random states
    if random_state is not None:
        random_seeds = np.random.RandomState(random_state).randint(
            np.iinfo(np.int32).max, size=n_bootstrap_iterations)
    else:
        random_seeds = [None] * n_bootstrap_iterations
    
    def _single_bootstrap(seed):
        sample = bootstrap_func(y, n_subsamples, seed)
        if isinstance(sample, tuple):
            return list(sample)
        return [sample]
    
    # Run bootstrap in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_bootstrap)(seed) for seed in random_seeds
    )
    
    # Flatten the results
    flattened = []
    for r in results:
        flattened.extend(r)
        
    return flattened
