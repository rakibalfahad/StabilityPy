"""
Benchmark Comparison: CPU vs GPU Performance for Stability Selection

This script benchmarks the performance of stability selection 
with and without GPU acceleration across different dataset sizes.
It generates a performance comparison plot and provides statistics
on the speedup factors.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from stability_selection import StabilitySelection


def generate_dataset(n_samples, n_features, n_informative=10, random_state=42):
    """Generate a synthetic dataset with specified dimensions."""
    rng = check_random_state(random_state)
    
    # Generate random features
    X = rng.normal(0, 1, size=(n_samples, n_features))
    
    # Select important features
    important_features = np.sort(rng.choice(np.arange(n_features), size=n_informative, replace=False))
    
    # Generate target (simple linear combination with noise)
    y = np.zeros(n_samples)
    for i in important_features:
        y += X[:, i] * rng.normal(0, 1)
    
    # Add noise
    y += rng.normal(0, 0.1, size=n_samples)
    
    # Convert to binary classification
    y = (y > 0).astype(int)
    
    return X, y, important_features


def benchmark_stability_selection(X, y, use_gpu=False, n_bootstrap=50, lambda_grid_size=25):
    """Run stability selection and time it."""
    # Create base estimator
    base_estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(penalty='l1', solver='liblinear'))
    ])
    
    # Configure lambda grid
    lambda_grid = np.logspace(-5, -1, lambda_grid_size)
    
    # Create and time the selector
    start_time = time.time()
    selector = StabilitySelection(
        base_estimator=base_estimator, 
        lambda_name='model__C',
        lambda_grid=lambda_grid,
        n_bootstrap_iterations=n_bootstrap,
        n_jobs=-1,
        use_gpu=use_gpu,
        verbose=0
    )
    selector.fit(X, y)
    elapsed_time = time.time() - start_time
    
    return elapsed_time, selector


def run_benchmarks():
    """Run benchmarks across different dataset sizes."""
    # Define dataset sizes to test
    sample_sizes = [500, 1000, 2000]
    feature_sizes = [1000, 2000, 5000]
    
    results = {
        'dataset_label': [],
        'cpu_time': [],
        'gpu_time': [],
        'speedup': []
    }
    
    for n_samples in sample_sizes:
        for n_features in feature_sizes:
            print(f"\nBenchmarking dataset with {n_samples} samples and {n_features} features")
            X, y, _ = generate_dataset(n_samples, n_features)
            
            # Run with CPU
            print("  Running with CPU...")
            cpu_time, _ = benchmark_stability_selection(X, y, use_gpu=False)
            print(f"  CPU time: {cpu_time:.2f} seconds")
            
            # Run with GPU
            print("  Running with GPU...")
            gpu_time, _ = benchmark_stability_selection(X, y, use_gpu=True)
            print(f"  GPU time: {gpu_time:.2f} seconds")
            
            # Calculate speedup
            speedup = cpu_time / gpu_time
            print(f"  Speedup: {speedup:.2f}x")
            
            # Store results
            results['dataset_label'].append(f"{n_samples}×{n_features}")
            results['cpu_time'].append(cpu_time)
            results['gpu_time'].append(gpu_time)
            results['speedup'].append(speedup)
    
    return results


def plot_results(results):
    """Create a visualization of the benchmark results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Dataset labels and indices
    labels = results['dataset_label']
    x = np.arange(len(labels))
    width = 0.35
    
    # Plot execution times
    cpu_bars = ax1.bar(x - width/2, results['cpu_time'], width, label='CPU', color='royalblue')
    gpu_bars = ax1.bar(x + width/2, results['gpu_time'], width, label='GPU', color='orangered')
    
    ax1.set_xlabel('Dataset Size (samples×features)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('CPU vs GPU Execution Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.legend()
    
    # Plot speedup factors
    ax2.bar(x, results['speedup'], width=0.6, color='green')
    ax2.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)
    
    ax2.set_xlabel('Dataset Size (samples×features)')
    ax2.set_ylabel('Speedup Factor (CPU time / GPU time)')
    ax2.set_title('GPU Acceleration Speedup')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    
    # Add speedup values as text
    for i, v in enumerate(results['speedup']):
        ax2.text(i, v + 0.1, f"{v:.2f}x", ha='center')
    
    plt.tight_layout()
    plt.savefig('gpu_benchmark_results.png')
    print("\nBenchmark results saved to 'gpu_benchmark_results.png'")
    plt.show()


if __name__ == "__main__":
    print("Starting GPU vs CPU performance benchmark...")
    results = run_benchmarks()
    plot_results(results)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average speedup: {np.mean(results['speedup']):.2f}x")
    print(f"Maximum speedup: {np.max(results['speedup']):.2f}x (on dataset {results['dataset_label'][np.argmax(results['speedup'])]})")
    print(f"Minimum speedup: {np.min(results['speedup']):.2f}x (on dataset {results['dataset_label'][np.argmin(results['speedup'])]})")
