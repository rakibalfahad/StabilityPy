"""
This example demonstrates stability selection with GPU acceleration if available.
It compares the performance between CPU and GPU implementations.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from stability_selection import StabilitySelection


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
    
    # Configure lambda grid
    lambda_grid = np.logspace(-5, -1, 50)
    
    # First, run on CPU
    print("\nRunning stability selection on CPU...")
    start_time = time.time()
    cpu_selector = StabilitySelection(
        base_estimator=base_estimator, 
        lambda_name='model__C',
        lambda_grid=lambda_grid,
        n_jobs=-1,  # Use all CPU cores
        use_gpu=False,
        verbose=1
    )
    cpu_selector.fit(X, y)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.2f} seconds")
    
    # Now try with GPU if available
    print("\nRunning stability selection with GPU acceleration if available...")
    start_time = time.time()
    gpu_selector = StabilitySelection(
        base_estimator=base_estimator, 
        lambda_name='model__C',
        lambda_grid=lambda_grid,
        n_jobs=-1,
        use_gpu=True,
        verbose=1
    )
    gpu_selector.fit(X, y)
    gpu_time = time.time() - start_time
    print(f"GPU time: {gpu_time:.2f} seconds")
    
    if cpu_time > gpu_time:
        speedup = cpu_time / gpu_time
        print(f"GPU provided a {speedup:.2f}x speedup")
    else:
        print("CPU was faster in this case")
    
    # Compare the selected features
    cpu_selected = cpu_selector.get_support(indices=True)
    gpu_selected = gpu_selector.get_support(indices=True)
    
    print("\nSelected feature indices (CPU):", cpu_selected)
    print("Selected feature indices (GPU):", gpu_selected)
    
    # Compare with true important features
    cpu_correct = np.isin(cpu_selected, important_betas).sum()
    gpu_correct = np.isin(gpu_selected, important_betas).sum()
    
    print(f"\nTrue positives - CPU: {cpu_correct}/{len(important_betas)}")
    print(f"True positives - GPU: {gpu_correct}/{len(important_betas)}")
    
    # Plot stability scores
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    paths_to_highlight = np.isin(np.arange(p), important_betas)
    x_grid = cpu_selector.lambda_grid / np.max(cpu_selector.lambda_grid)
    
    plt.plot(x_grid, cpu_selector.stability_scores_[~paths_to_highlight].T, 'k:', linewidth=0.5, alpha=0.3)
    plt.plot(x_grid, cpu_selector.stability_scores_[paths_to_highlight].T, 'r-', linewidth=2)
    plt.plot(x_grid, cpu_selector.threshold * np.ones_like(cpu_selector.lambda_grid), 'b--', linewidth=1)
    plt.xlabel('Lambda / max(Lambda)')
    plt.ylabel('Stability score')
    plt.title('CPU Stability Paths')
    
    plt.subplot(1, 2, 2)
    plt.plot(x_grid, gpu_selector.stability_scores_[~paths_to_highlight].T, 'k:', linewidth=0.5, alpha=0.3)
    plt.plot(x_grid, gpu_selector.stability_scores_[paths_to_highlight].T, 'r-', linewidth=2)
    plt.plot(x_grid, gpu_selector.threshold * np.ones_like(gpu_selector.lambda_grid), 'b--', linewidth=1)
    plt.xlabel('Lambda / max(Lambda)')
    plt.ylabel('Stability score')
    plt.title('GPU Stability Paths')
    
    plt.tight_layout()
    plt.savefig('stability_paths_comparison.png')
    print("\nStability paths comparison saved to 'stability_paths_comparison.png'")
    plt.show()
