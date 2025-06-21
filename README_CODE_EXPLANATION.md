# StabilityPy Code Explanation

This document provides detailed explanations of the code in the StabilityPy project, with examples for each component.

## Table of Contents

1. [Core Stability Selection](#1-core-stability-selection)
2. [Data Processing](#2-data-processing)
3. [Feature Selection](#3-feature-selection)
4. [Model Fine-tuning](#4-model-fine-tuning)
5. [Visualization](#5-visualization)
6. [Synthetic Data Generation](#6-synthetic-data-generation)
7. [GPU Acceleration](#7-gpu-acceleration)
8. [Complete Workflow Example](#8-complete-workflow-example)

## 1. Core Stability Selection

The core of StabilityPy is the `StabilitySelection` class, which implements the stability selection algorithm.

### Key Components

```python
from stability_selection import StabilitySelection
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create a base estimator
base_estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty='l1', solver='liblinear'))
])

# Initialize stability selection
selector = StabilitySelection(
    base_estimator=base_estimator,
    lambda_name='model__C',  # Parameter to tune for regularization
    lambda_grid=np.logspace(-5, 0, 25),  # Range of regularization values
    threshold=0.7,  # Threshold for selection
    n_bootstrap_iterations=100,  # Number of bootstrap samples
    bootstrap_func='subsample',  # Bootstrap method
    n_jobs=-1,  # Use all CPU cores
    use_gpu=False  # Whether to use GPU acceleration
)

# Fit the selector to data
selector.fit(X, y)

# Get selected features
selected_features = selector.get_support(indices=True)
```

### How It Works

1. **Bootstrap Sampling**: The algorithm creates multiple subsamples of the data.
2. **Parameter Grid**: For each subsample, it fits the base estimator with different regularization values.
3. **Stability Scores**: It calculates how frequently each feature is selected across all subsamples and parameters.
4. **Thresholding**: Features with stability scores above the threshold are considered stable and selected.

## 2. Data Processing

The `stability_processor.py` script handles data loading, preprocessing, and analysis.

### Loading Data

```python
def load_data(file_path):
    """Load data from CSV or CSV.GZ file"""
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.csv.gz') or file_path.endswith('.gz'):
        data = pd.read_csv(file_path, compression='gzip')
    
    # Separate features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    return X, y
```

### Preprocessing

```python
def preprocess_data(X, y, problem_type):
    """Preprocess data for stability selection"""
    # Store feature names
    feature_names = X.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Process target based on problem type
    if problem_type == 'classification':
        # Encode categorical target if needed
        if y.dtype == 'object':
            encoder = LabelEncoder()
            y_processed = encoder.fit_transform(y)
        else:
            y_processed = y.values
    else:  # regression
        y_processed = y.values
    
    return X_scaled, y_processed, feature_names
```

### Example

```bash
# Process a classification dataset
python stability_processor.py --input data.csv --output results_dir --problem_type classification

# Process a regression dataset
python stability_processor.py --input regression_data.csv.gz --output regression_results --problem_type regression --use_gpu
```

## 3. Feature Selection

StabilityPy provides robust feature selection with control over false positives.

### Running Stability Selection

```python
def run_stability_selection(X, y, feature_names, problem_type, output_dir, use_gpu=False, n_bootstrap=100):
    """Run stability selection on the data"""
    # Configure base estimator based on problem type
    if problem_type == 'classification':
        base_estimator = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(penalty='l1', solver='liblinear'))
        ])
        lambda_name = 'model__C'
        lambda_grid = np.logspace(-5, 0, 25)  # C values for LogisticRegression
    else:  # regression
        base_estimator = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso())
        ])
        lambda_name = 'model__alpha'
        lambda_grid = np.logspace(-5, 0, 25)  # alpha values for Lasso
    
    # Run stability selection
    selector = StabilitySelection(
        base_estimator=base_estimator,
        lambda_name=lambda_name,
        lambda_grid=lambda_grid,
        threshold=0.7,
        n_bootstrap_iterations=n_bootstrap,
        bootstrap_func='subsample',
        n_jobs=-1,
        use_gpu=use_gpu
    )
    
    selector.fit(X, y)
    
    # Get selected features
    support = selector.get_support()
    selected_indices = np.where(support)[0]
    selected_features = [feature_names[i] for i in selected_indices]
    
    return selector, selected_features
```

### Example Output

```
Running stability selection with 100 bootstrap iterations
GPU acceleration: disabled
Fitting models: 100%|██████████| 25/25 [00:15<00:00, 1.67it/s]
Selected 8 features: ['feature_2', 'feature_7', 'feature_9', 'feature_12', 'feature_15', 'feature_23', 'feature_45', 'feature_67']
```

## 4. Model Fine-tuning

After selecting features, StabilityPy can fine-tune models using only the selected features.

### Fine-tuning Process

```python
def fine_tune_model(X, y, selected_features, feature_names, problem_type, output_dir):
    """Fine-tune a model using the selected features"""
    # Get indices of selected features
    selected_indices = [feature_names.index(feature) for feature in selected_features]
    
    # Split data into train and test
    X_selected = X[:, selected_indices]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    
    # Also create a baseline with all features
    X_all_train, X_all_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Choose model based on problem type
    if problem_type == 'classification':
        # Model with selected features
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Baseline model with all features
        baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
        baseline_model.fit(X_all_train, y_train)
        baseline_pred = baseline_model.predict(X_all_test)
        baseline_accuracy = accuracy_score(y_test, baseline_pred)
        baseline_f1 = f1_score(y_test, baseline_pred, average='weighted')
        
        performance = {
            'selected_accuracy': accuracy,
            'selected_f1': f1,
            'baseline_accuracy': baseline_accuracy,
            'baseline_f1': baseline_f1
        }
```

### Example Output

```
Fine-tuning model with 8 selected features
Selected features - Accuracy: 0.9233, F1 Score: 0.9225
All features - Accuracy: 0.9167, F1 Score: 0.9152
```

## 5. Visualization

StabilityPy creates multiple visualizations to help interpret the results.

### Stability Paths

```python
# Create stability paths plot
plt.figure(figsize=(12, 8))

# Plot non-selected features
for i in range(len(feature_names)):
    if i not in selected_indices:
        plt.plot(lambda_grid_normalized, selector.stability_scores_[i, :], 
                 'k-', alpha=0.1)

# Plot selected features
for i in selected_indices:
    plt.plot(lambda_grid_normalized, selector.stability_scores_[i, :], 
             'r-', linewidth=2, label=feature_names[i])

# Plot threshold
plt.axhline(y=selector.threshold, color='b', linestyle='--', 
            label=f'Threshold ({selector.threshold})')

plt.xlabel('λ / max(λ)')
plt.ylabel('Stability Score')
plt.title('Stability Selection Paths')
```

### Performance Comparison

```python
# Create bar plot of performance metrics
plt.figure(figsize=(10, 6))

if problem_type == 'classification':
    metrics = ['Accuracy', 'F1 Score']
    selected_values = [performance['selected_accuracy'], performance['selected_f1']]
    baseline_values = [performance['baseline_accuracy'], performance['baseline_f1']]
else:  # regression
    metrics = ['R²', 'MSE (normalized)']
    selected_values = [performance['selected_r2'], 1 - performance['selected_mse'] / performance['baseline_mse']]
    baseline_values = [performance['baseline_r2'], 1 - performance['baseline_mse'] / performance['baseline_mse']]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, selected_values, width, label='Selected Features')
plt.bar(x + width/2, baseline_values, width, label='All Features')
```

## 6. Synthetic Data Generation

The `synthetic_data_generator.py` script creates controlled datasets for testing.

### Classification Data

```python
def generate_classification_data(n_samples=1000, n_features=100, n_informative=10,
                                n_redundant=10, n_repeated=0, n_classes=2,
                                random_state=42, add_noise=0.1):
    """Generate synthetic classification data"""
    # Generate data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        random_state=random_state,
        shuffle=True
    )
    
    # Add noise
    if add_noise > 0:
        rng = check_random_state(random_state)
        noise = rng.normal(0, add_noise, X.shape)
        X = X + noise
    
    return X, y, feature_indices
```

### Regression Data

```python
def generate_regression_data(n_samples=1000, n_features=100, n_informative=10,
                           effective_rank=None, tail_strength=0.5, noise=0.1,
                           random_state=42, add_noise=0.1):
    """Generate synthetic regression data"""
    # Generate data
    X, y, coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        effective_rank=effective_rank,
        tail_strength=tail_strength,
        noise=noise,
        random_state=random_state,
        coef=True
    )
    
    # Add noise
    if add_noise > 0:
        rng = check_random_state(random_state)
        noise_matrix = rng.normal(0, add_noise, X.shape)
        X = X + noise_matrix
    
    return X, y, feature_indices
```

### Example

```bash
# Generate a classification dataset
python synthetic_data_generator.py --output class_data.csv --problem_type classification --n_samples 1000 --n_features 100 --n_informative 10

# Generate a regression dataset
python synthetic_data_generator.py --output reg_data.csv.gz --problem_type regression --n_samples 2000 --n_features 500 --n_informative 20 --compress
```

## 7. GPU Acceleration

StabilityPy supports GPU acceleration via PyTorch to speed up computations.

### Enabling GPU Acceleration

```python
selector = StabilitySelection(
    base_estimator=base_estimator,
    lambda_name='model__C',
    lambda_grid=np.logspace(-5, 0, 25),
    threshold=0.7,
    use_gpu=True  # Enable GPU acceleration
)
```

### How It Works

```python
def _check_gpu_availability():
    """Check if PyTorch is available and if a CUDA-compatible GPU is available."""
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()

# In StabilitySelection.fit()
if self.use_gpu:
    gpu_available = _check_gpu_availability()
    if not gpu_available:
        warnings.warn("GPU requested but not available. Falling back to CPU.")
        self.use_gpu = False
```

### Example with Performance Comparison

```python
# First, run on CPU
start_time = time.time()
cpu_selector = StabilitySelection(
    base_estimator=base_estimator, 
    lambda_name='model__C',
    lambda_grid=lambda_grid,
    n_jobs=-1,
    use_gpu=False
)
cpu_selector.fit(X, y)
cpu_time = time.time() - start_time

# Now try with GPU
start_time = time.time()
gpu_selector = StabilitySelection(
    base_estimator=base_estimator, 
    lambda_name='model__C',
    lambda_grid=lambda_grid,
    n_jobs=-1,
    use_gpu=True
)
gpu_selector.fit(X, y)
gpu_time = time.time() - start_time

speedup = cpu_time / gpu_time
print(f"GPU provided a {speedup:.2f}x speedup")
```

## 8. Complete Workflow Example

This example shows a complete workflow from data generation to model fine-tuning.

```bash
# 1. Generate synthetic data
python synthetic_data_generator.py \
    --output data/synthetic_classification.csv \
    --problem_type classification \
    --n_samples 1000 \
    --n_features 100 \
    --n_informative 10 \
    --noise 0.1

# 2. Run stability selection with fine-tuning
python stability_processor.py \
    --input data/synthetic_classification.csv \
    --output results/synthetic_classification \
    --problem_type classification \
    --n_bootstrap 100 \
    --use_gpu
```

### Typical Output Structure

```
results/synthetic_classification/
├── stability_selection_results.pkl  # Saved stability selection results
├── stability_selector.pkl           # Saved selector object
├── selected_features.csv            # CSV of selected features
├── feature_importance.csv           # Feature importances from fine-tuned model
├── fine_tuned_model.pkl             # Fine-tuned model
├── baseline_model.pkl               # Baseline model
├── performance_metrics.csv          # Performance comparison metrics
├── stability_paths.png              # Stability paths visualization
├── stability_heatmap.png            # Heatmap of stability scores
├── performance_comparison.png       # Bar plot of performance metrics
└── feature_importance.png           # Bar plot of feature importances
```
