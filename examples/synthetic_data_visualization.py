"""
Synthetic Data Generation and Stability Selection Visualization

This script:
1. Generates synthetic data with interdependent features based on a mathematical model
2. Applies stability selection to identify the relevant features
3. Creates visualizations:
   - Histogram of stability scores
   - Stability paths (as shown in the original paper)
   - Heatmap of feature correlations
   - Comparison of true vs selected features
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from stability_selection import StabilitySelection, RandomizedLasso, plot_stability_path


def generate_synthetic_data(n_samples=200, n_features=1000, n_informative=10, 
                           equation_type='friedman', noise_level=0.1, random_state=42):
    """
    Generate synthetic data with interdependent features based on popular equations.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Total number of features
    n_informative : int
        Number of informative features
    equation_type : str
        Type of equation to use: 'friedman', 'polynomial', or 'sinusoidal'
    noise_level : float
        Level of noise to add
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X : ndarray, shape (n_samples, n_features)
        The generated feature matrix
    y : ndarray, shape (n_samples,)
        The target values
    relevant_features : ndarray
        Indices of the truly relevant features
    """
    rng = check_random_state(random_state)
    
    # Generate random features
    X = rng.normal(0, 1, size=(n_samples, n_features))
    
    # Create some non-linear relationships between features
    for i in range(1, n_informative // 2):
        factor = rng.uniform(0.8, 1.2)
        X[:, i] = factor * X[:, 0] + rng.normal(0, 0.1, size=n_samples)
    
    # Select the truly relevant features
    relevant_features = np.sort(rng.choice(np.arange(n_features), size=n_informative, replace=False))
    
    # Generate target variable based on equation type
    if equation_type == 'friedman':
        # Friedman's equation (popular in ML benchmarking)
        # y = 10*sin(π*x₁*x₂) + 20*(x₃-0.5)² + 10*x₄ + 5*x₅ + ε
        X_rel = X[:, relevant_features]
        y = (10 * np.sin(np.pi * X_rel[:, 0] * X_rel[:, 1]) + 
             20 * np.square(X_rel[:, 2] - 0.5) + 
             10 * X_rel[:, 3] + 
             5 * X_rel[:, 4])
        
        # Add interactions between features
        for i in range(5, n_informative):
            if i < len(relevant_features):
                y += X_rel[:, i] * X_rel[:, i % 5]
                
    elif equation_type == 'polynomial':
        # Polynomial relationship
        X_rel = X[:, relevant_features]
        y = (X_rel[:, 0]**2 + 
             2 * X_rel[:, 1] * X_rel[:, 2] + 
             X_rel[:, 3]**3 - 
             np.abs(X_rel[:, 4]))
        
        # Add additional polynomial terms
        for i in range(5, n_informative):
            if i < len(relevant_features):
                y += 0.5 * X_rel[:, i]**2
                
    elif equation_type == 'sinusoidal':
        # Sinusoidal relationship
        X_rel = X[:, relevant_features]
        y = (np.sin(X_rel[:, 0]) + 
             np.cos(X_rel[:, 1]) + 
             np.sin(X_rel[:, 2] * X_rel[:, 3]) + 
             X_rel[:, 4])
        
        # Add additional sinusoidal terms
        for i in range(5, n_informative):
            if i < len(relevant_features):
                y += 0.5 * np.sin(2 * X_rel[:, i])
    else:
        raise ValueError(f"Unknown equation type: {equation_type}")
    
    # Add noise
    y += rng.normal(0, noise_level * np.std(y), size=n_samples)
    
    return X, y, relevant_features


def plot_stability_score_histogram(stability_scores, threshold=None, ax=None):
    """
    Plot histogram of stability scores.
    
    Parameters:
    -----------
    stability_scores : ndarray
        Stability scores for each feature
    threshold : float, optional
        Threshold for feature selection
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    max_scores = np.max(stability_scores, axis=1)
    
    # Plot histogram
    sns.histplot(max_scores, bins=20, kde=True, ax=ax)
    
    if threshold is not None:
        ax.axvline(threshold, color='r', linestyle='--', 
                   label=f'Threshold ({threshold})')
        
    ax.set_xlabel('Maximum Stability Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Stability Scores Across Features')
    
    # Add count of features above threshold
    if threshold is not None:
        n_selected = np.sum(max_scores > threshold)
        ax.text(0.98, 0.95, f'Selected: {n_selected}', 
                horizontalalignment='right',
                verticalalignment='top', 
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
    
    return ax


def plot_correlation_heatmap(X, relevant_features, selected_features, ax=None):
    """
    Plot heatmap of feature correlations.
    
    Parameters:
    -----------
    X : ndarray
        The feature matrix
    relevant_features : ndarray
        Indices of the truly relevant features
    selected_features : ndarray
        Indices of the selected features
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
    # Combine relevant and selected features
    all_features = np.unique(np.concatenate([relevant_features, selected_features]))
    
    if len(all_features) > 20:
        print("Too many features for heatmap, showing only the top 20")
        all_features = all_features[:20]
        
    # Calculate correlation matrix
    X_subset = X[:, all_features]
    corr = np.corrcoef(X_subset.T)
    
    # Create labels
    labels = [f"F{i}" for i in all_features]
    
    # Mark true relevant features
    for i, feature in enumerate(all_features):
        if feature in relevant_features:
            labels[i] = f"F{feature}*"
            
    # Plot heatmap
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_title('Correlation Between Relevant and Selected Features')
    
    return ax


def plot_feature_importance_comparison(relevant_features, selected_features, 
                                      n_features, stability_scores=None, ax=None):
    """
    Plot comparison of true vs. selected features.
    
    Parameters:
    -----------
    relevant_features : ndarray
        Indices of the truly relevant features
    selected_features : ndarray
        Indices of the selected features
    n_features : int
        Total number of features
    stability_scores : ndarray, optional
        Stability scores for each feature
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        
    # Create feature status array
    feature_status = np.zeros(n_features)
    # Mark true relevant features as 1
    feature_status[relevant_features] = 1
    # Mark selected features as 2
    feature_status[selected_features] = 2
    # Mark features that are both relevant and selected as 3
    feature_status[np.intersect1d(relevant_features, selected_features)] = 3
    
    # Calculate metrics
    true_positives = np.sum(feature_status == 3)
    false_positives = np.sum(feature_status == 2)
    false_negatives = np.sum(feature_status == 1)
    
    # For visualization, filter to show only the relevant ones
    show_features = np.union1d(relevant_features, selected_features)
    # Add some context by including neighboring features
    neighbors = np.concatenate([show_features - 1, show_features + 1])
    show_features = np.union1d(show_features, neighbors)
    show_features = show_features[(show_features >= 0) & (show_features < n_features)]
    
    # Sort for better visualization
    show_features = np.sort(show_features)
    
    # Create bar colors
    colors = []
    bar_heights = []
    for i in show_features:
        status = feature_status[i]
        if status == 0:  # Not relevant, not selected
            colors.append('lightgrey')
            bar_heights.append(0.2)  # Small bar for context
        elif status == 1:  # Relevant but not selected (false negative)
            colors.append('orange')
            bar_heights.append(0.7)
        elif status == 2:  # Selected but not relevant (false positive)
            colors.append('red')
            bar_heights.append(0.7)
        else:  # Both relevant and selected (true positive)
            colors.append('green')
            bar_heights.append(1.0)
            
    # Plot bars
    x_pos = np.arange(len(show_features))
    bars = ax.bar(x_pos, bar_heights, color=colors)
    
    # If stability scores are provided, plot them as a line
    if stability_scores is not None:
        max_scores = np.max(stability_scores, axis=1)
        scores_to_plot = max_scores[show_features]
        ax_twin = ax.twinx()
        ax_twin.plot(x_pos, scores_to_plot, 'k--', alpha=0.7, label='Stability Score')
        ax_twin.set_ylabel('Stability Score')
        ax_twin.set_ylim(0, 1.1)
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{i}" for i in show_features], rotation=90)
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='True Positive'),
        Patch(facecolor='orange', label='False Negative'),
        Patch(facecolor='red', label='False Positive')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add metrics as text
    metrics_text = (f"True Positives: {true_positives}/{len(relevant_features)}\n"
                   f"False Positives: {false_positives}\n"
                   f"False Negatives: {false_negatives}")
    ax.text(0.02, 0.95, metrics_text, 
            horizontalalignment='left',
            verticalalignment='top', 
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Feature Status')
    ax.set_title('Comparison of True Relevant Features vs. Selected Features')
    
    return ax


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    n_samples = 200
    n_features = 1000
    n_informative = 10
    equation_type = 'friedman'  # 'friedman', 'polynomial', or 'sinusoidal'
    
    print(f"Generating synthetic data with {n_samples} samples and {n_features} features...")
    print(f"Using {equation_type} equation with {n_informative} informative features")
    
    # Generate synthetic data
    X, y, relevant_features = generate_synthetic_data(
        n_samples=n_samples, 
        n_features=n_features,
        n_informative=n_informative,
        equation_type=equation_type
    )
    
    print(f"True relevant features: {relevant_features}")
    
    # Create base estimator for stability selection
    if equation_type == 'friedman' or equation_type == 'polynomial':
        # For regression problems
        base_estimator = RandomizedLasso(alpha=0.1, weakness=0.5)
        lambda_name = 'alpha'
        lambda_grid = np.logspace(-3, 0, 30)
    else:
        # For classification, convert target to binary
        y_binary = (y > np.median(y)).astype(int)
        y = y_binary
        
        # Create a pipeline with standardization and logistic regression
        base_estimator = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(penalty='l1', solver='liblinear'))
        ])
        lambda_name = 'model__C'
        lambda_grid = np.logspace(-5, -1, 30)
    
    # Set up stability selection
    selector = StabilitySelection(
        base_estimator=base_estimator,
        lambda_name=lambda_name,
        lambda_grid=lambda_grid,
        threshold=0.6,
        n_bootstrap_iterations=100,
        sample_fraction=0.75,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit stability selection
    print("\nFitting stability selection...")
    selector.fit(X, y)
    
    # Get selected features
    selected_features = selector.get_support(indices=True)
    selected_scores = selector.stability_scores_.max(axis=1)[selected_features]
    
    print("\nSelected features:")
    for idx, (feature, score) in enumerate(zip(selected_features, selected_scores)):
        print(f"Feature {idx + 1}: [index: {feature}], score: {score:.3f}")
    
    # Calculate metrics
    true_positives = np.intersect1d(relevant_features, selected_features)
    false_positives = np.setdiff1d(selected_features, relevant_features)
    false_negatives = np.setdiff1d(relevant_features, selected_features)
    
    print(f"\nTrue positives: {len(true_positives)}/{len(relevant_features)} "
          f"({100 * len(true_positives) / len(relevant_features):.1f}%)")
    print(f"False positives: {len(false_positives)}")
    print(f"False negatives: {len(false_negatives)}")
    
    # Create visualizations
    plt.figure(figsize=(16, 14))
    
    # 1. Stability path (as in original paper)
    plt.subplot(2, 2, 1)
    ax1 = plt.gca()
    x_grid = selector.lambda_grid / np.max(selector.lambda_grid)
    
    # Plot all paths with low alpha
    ax1.plot(x_grid, selector.stability_scores_.T, 'k-', linewidth=0.5, alpha=0.1)
    
    # Highlight true relevant features
    for feature in relevant_features:
        ax1.plot(x_grid, selector.stability_scores_[feature], 'g-', linewidth=2, alpha=0.7)
    
    # Highlight threshold
    ax1.plot(x_grid, selector.threshold * np.ones_like(selector.lambda_grid), 'r--', linewidth=1)
    
    ax1.set_xlabel('λ / max(λ)')
    ax1.set_ylabel('Stability Score')
    ax1.set_title('Stability Paths (Green = True Relevant Features)')
    
    # 2. Histogram of stability scores
    plt.subplot(2, 2, 2)
    ax2 = plot_stability_score_histogram(selector.stability_scores_, selector.threshold)
    
    # 3. Correlation heatmap
    plt.subplot(2, 2, 3)
    ax3 = plot_correlation_heatmap(X, relevant_features, selected_features)
    
    # 4. Feature importance comparison
    plt.subplot(2, 2, 4)
    ax4 = plot_feature_importance_comparison(
        relevant_features, selected_features, n_features, selector.stability_scores_)
    
    plt.tight_layout()
    plt.savefig('stability_selection_visualization.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to 'stability_selection_visualization.png'")
    
    # Show additional plot similar to original paper
    plt.figure(figsize=(10, 6))
    paths_to_highlight = np.zeros(n_features, dtype=bool)
    paths_to_highlight[relevant_features] = True
    
    # Plot non-relevant features with low alpha
    plt.plot(x_grid, selector.stability_scores_[~paths_to_highlight].T, 
             'k:', linewidth=0.5, alpha=0.1)
    
    # Plot relevant features with high alpha
    plt.plot(x_grid, selector.stability_scores_[paths_to_highlight].T, 
             'r-', linewidth=1.5)
    
    # Add threshold line
    plt.plot(x_grid, selector.threshold * np.ones_like(selector.lambda_grid), 
             'b--', linewidth=1)
    
    plt.xlabel('λ / max(λ)')
    plt.ylabel('Stability Score')
    plt.title('Stability Paths (Original Paper Style)')
    plt.savefig('stability_paths_paper_style.png', dpi=300, bbox_inches='tight')
    print("Paper-style stability paths saved to 'stability_paths_paper_style.png'")
    
    plt.show()


if __name__ == "__main__":
    main()
