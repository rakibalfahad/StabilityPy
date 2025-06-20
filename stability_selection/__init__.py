"""
Stability Selection module.

This module implements stability selection, a method for feature selection based on
subsampling and selection algorithms like LASSO.
"""

from .bootstrap import (bootstrap_without_replacement,
                        complementary_pairs_bootstrap,
                        stratified_bootstrap)
from .randomized_lasso import (RandomizedLogisticRegression,
                             RandomizedLasso)
from .stability_selection import StabilitySelection, plot_stability_path

__all__ = ['StabilitySelection',
           'RandomizedLogisticRegression',
           'RandomizedLasso',
           'bootstrap_without_replacement',
           'complementary_pairs_bootstrap',
           'stratified_bootstrap',
           'plot_stability_path']
