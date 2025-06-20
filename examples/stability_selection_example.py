"""
This example demonstrates the basic usage of stability selection.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from stability_selection import StabilitySelection, plot_stability_path


def _generate_dummy_classification_data(p=1000, n=1000, k=5, random_state=123321):
    """Generate synthetic classification data with known important features."""
    rng = check_random_state(random_state)

    X = rng.normal(loc=0.0, scale=1.0, size=(n, p))
    betas = np.zeros(p)
    important_betas = np.sort(rng.choice(a=np.arange(p), size=k))
    betas[important_betas] = rng.uniform(size=k)

    probs = 1 / (1 + np.exp(-1 * np.matmul(X, betas)))
    y = (probs > 0.5).astype(int)

    return X, y, important_betas


if __name__ == '__main__':
    n, p, k = 500, 1000, 5  # Sample size, dimensionality, number of important features
    
    print("Generating synthetic data...")
    X, y, important_betas = _generate_dummy_classification_data(n=n, p=p, k=k)
    print(f"Generated data: {X.shape}, with important features at indices: {important_betas}")
    
    # Create a base estimator pipeline
    base_estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(penalty='l1', solver='liblinear'))
    ])
    
    # Run stability selection
    print("\nRunning stability selection...")
    selector = StabilitySelection(
        base_estimator=base_estimator, 
        lambda_name='model__C',
        lambda_grid=np.logspace(-5, -1, 50),
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    selector.fit(X, y)
    
    # Get selected features
    selected_variables = selector.get_support(indices=True)
    selected_scores = selector.stability_scores_.max(axis=1)[selected_variables]
    
    print("\nSelected features:")
    print("-----------------")
    for idx, (variable, score) in enumerate(zip(selected_variables, selected_scores)):
        print(f"Feature {idx + 1}: [index: {variable}], score: {score:.3f}")
    
    # Calculate how many true features were recovered
    true_positives = np.isin(selected_variables, important_betas).sum()
    print(f"\nRecovered {true_positives}/{len(important_betas)} true important features")
    
    # Plot stability path
    print("\nPlotting stability path...")
    fig, ax = plot_stability_path(selector)
    
    # Highlight the true important features
    x_grid = selector.lambda_grid / np.max(selector.lambda_grid)
    for beta in important_betas:
        ax.plot(x_grid, selector.stability_scores_[beta], 'g-', linewidth=2,
                label=f"True important feature (index {beta})")
        # Only add the label once to avoid duplicate legend entries
        ax.get_lines()[-1].set_label('_nolegend_' if beta != important_betas[0] else "True important feature")
    
    # Add legend
    ax.legend()
    
    # Save the figure
    plt.savefig('stability_path.png')
    print("Stability path saved to 'stability_path.png'")
    plt.show()
