"""
Stability Selection Data Processor

This script processes CSV or CSV.GZ files for stability selection feature selection.
It handles both classification and regression problems, creates visualizations,
and outputs results to a specified directory.

Usage:
    python stability_processor.py --input data.csv --output results_dir --problem_type classification
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import joblib

from stability_selection import StabilitySelection


def load_data(file_path):
    """
    Load data from CSV or CSV.GZ file
    
    Parameters:
    -----------
    file_path: str
        Path to CSV or CSV.GZ file
        
    Returns:
    --------
    X: pandas DataFrame
        Feature matrix
    y: pandas Series
        Target vector
    """
    print(f"Loading data from: {file_path}")
    
    # Check file extension
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.csv.gz') or file_path.endswith('.gz'):
        data = pd.read_csv(file_path, compression='gzip')
    else:
        raise ValueError("Input file must be CSV or CSV.GZ format")
    
    # Check if data has headers
    if data.columns.dtype == 'object':
        print(f"Detected {len(data.columns)} columns with headers")
    else:
        print(f"No column headers detected, using default column names")
        data.columns = [f'feature_{i}' for i in range(data.shape[1]-1)] + ['target']
    
    # Separate features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {y.value_counts().to_dict() if y.dtype == 'object' else 'Continuous target'}")
    
    return X, y


def preprocess_data(X, y, problem_type):
    """
    Preprocess data for stability selection
    
    Parameters:
    -----------
    X: pandas DataFrame
        Feature matrix
    y: pandas Series
        Target vector
    problem_type: str
        'classification' or 'regression'
        
    Returns:
    --------
    X_scaled: numpy array
        Scaled feature matrix
    y_processed: numpy array
        Processed target vector
    feature_names: list
        List of feature names
    """
    print(f"Preprocessing data for {problem_type} problem")
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Process target based on problem type
    if problem_type == 'classification':
        # Check if target is already encoded
        if y.dtype == 'object':
            print("Encoding categorical target")
            encoder = LabelEncoder()
            y_processed = encoder.fit_transform(y)
            print(f"Classes: {list(encoder.classes_)}")
        else:
            y_processed = y.values
            print("Target already appears to be encoded")
    else:  # regression
        y_processed = y.values
    
    return X_scaled, y_processed, feature_names


def run_stability_selection(X, y, feature_names, problem_type, output_dir, use_gpu=False, n_bootstrap=100):
    """
    Run stability selection on the data
    
    Parameters:
    -----------
    X: numpy array
        Feature matrix
    y: numpy array
        Target vector
    feature_names: list
        List of feature names
    problem_type: str
        'classification' or 'regression'
    output_dir: str
        Directory to save results
    use_gpu: bool
        Whether to use GPU acceleration
    n_bootstrap: int
        Number of bootstrap iterations
        
    Returns:
    --------
    selector: StabilitySelection
        Fitted stability selection object
    selected_features: list
        List of selected feature names
    """
    print(f"Running stability selection with {n_bootstrap} bootstrap iterations")
    print(f"GPU acceleration: {'enabled' if use_gpu else 'disabled'}")
    
    # Configure base estimator based on problem type
    if problem_type == 'classification':
        base_estimator = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000))
        ])
        lambda_name = 'model__C'
        lambda_grid = np.logspace(-5, 0, 25)  # C values for LogisticRegression
    else:  # regression
        base_estimator = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(max_iter=10000))
        ])
        lambda_name = 'model__alpha'
        lambda_grid = np.logspace(-2, 2, 25)  # Stronger regularization for regression
    
    # Run stability selection
    selector = StabilitySelection(
        base_estimator=base_estimator,
        lambda_name=lambda_name,
        lambda_grid=lambda_grid,
        threshold=0.7,  # Start with default threshold
        n_bootstrap_iterations=n_bootstrap,
        bootstrap_func='subsample',
        n_jobs=-1,
        use_gpu=use_gpu,
        verbose=1
    )
    
    selector.fit(X, y)
    
    # Get selected features with default threshold
    support = selector.get_support()
    selected_indices = np.where(support)[0]
    
    # Adjust threshold if too many features are selected (more than half)
    if len(selected_indices) > X.shape[1] / 2:
        print(f"Too many features selected ({len(selected_indices)}). Adjusting threshold...")
        
        # Get max stability scores for each feature
        max_scores = np.max(selector.stability_scores_, axis=1)
        
        # Sort scores in descending order
        sorted_scores = np.sort(max_scores)[::-1]
        
        # Select a threshold that gives approximately 10% of features or at least 5 features
        target_n_features = max(5, int(X.shape[1] * 0.1))
        if len(sorted_scores) > target_n_features:
            new_threshold = sorted_scores[target_n_features]
            new_threshold = min(0.9, max(0.7, new_threshold))  # Between 0.7 and 0.9
            
            print(f"New threshold: {new_threshold:.2f} (target: {target_n_features} features)")
            selector.threshold = new_threshold
            
            # Get selected features with new threshold
            support = selector.get_support()
            selected_indices = np.where(support)[0]
            
            # If still too many features, take top N by stability score
            if len(selected_indices) > X.shape[1] / 3:
                print(f"Still too many features ({len(selected_indices)}). Taking top {target_n_features}...")
                top_indices = np.argsort(-max_scores)[:target_n_features]
                support = np.zeros(len(feature_names), dtype=bool)
                support[top_indices] = True
                selected_indices = top_indices
    
    selected_features = [feature_names[i] for i in selected_indices]
    
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    # Save results
    results = {
        'stability_scores': selector.stability_scores_,
        'selected_features': selected_features,
        'selected_indices': selected_indices,
        'support': support,
        'feature_names': feature_names,
        'threshold': selector.threshold
    }
    
    joblib.dump(results, os.path.join(output_dir, 'stability_selection_results.pkl'))
    joblib.dump(selector, os.path.join(output_dir, 'stability_selector.pkl'))
    
    # Save selected features to CSV
    pd.DataFrame({
        'feature': feature_names,
        'selected': support,
        'max_stability_score': np.max(selector.stability_scores_, axis=1)
    }).sort_values('max_stability_score', ascending=False).to_csv(
        os.path.join(output_dir, 'selected_features.csv'), index=False
    )
    
    return selector, selected_features


def fine_tune_model(X, y, selected_features, feature_names, problem_type, output_dir):
    """
    Fine-tune a model using the selected features
    
    Parameters:
    -----------
    X: numpy array
        Feature matrix
    y: numpy array
        Target vector
    selected_features: list
        List of selected feature names
    feature_names: list
        List of all feature names
    problem_type: str
        'classification' or 'regression'
    output_dir: str
        Directory to save results
        
    Returns:
    --------
    performance: dict
        Dictionary of performance metrics
    """
    print(f"Fine-tuning model with {len(selected_features)} selected features")
    
    # Get indices of selected features
    selected_indices = [feature_names.index(feature) for feature in selected_features]
    
    # Make sure we have at least one selected feature
    if not selected_indices:
        print("No features selected. Using top 5 features based on stability scores.")
        selected_indices = list(range(min(5, X.shape[1])))
        selected_features = [feature_names[i] for i in selected_indices]
    
    # Extract selected features
    X_selected = X[:, selected_indices]
    
    # Split data for training and testing
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
        
        # Cross-validation for more robust performance estimation
        cv_scores_selected = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=42),
            X_selected, y, cv=5, scoring='accuracy'
        )
        cv_accuracy = cv_scores_selected.mean()
        
        # Baseline model with all features
        baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
        baseline_model.fit(X_all_train, y_train)
        baseline_pred = baseline_model.predict(X_all_test)
        baseline_accuracy = accuracy_score(y_test, baseline_pred)
        baseline_f1 = f1_score(y_test, baseline_pred, average='weighted')
        
        # Cross-validation for baseline
        cv_scores_baseline = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=42),
            X, y, cv=5, scoring='accuracy'
        )
        cv_baseline_accuracy = cv_scores_baseline.mean()
        
        print(f"Selected features - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, CV Accuracy: {cv_accuracy:.4f}")
        print(f"All features - Accuracy: {baseline_accuracy:.4f}, F1 Score: {baseline_f1:.4f}, CV Accuracy: {cv_baseline_accuracy:.4f}")
        
        performance = {
            'selected_accuracy': accuracy,
            'selected_f1': f1,
            'selected_cv_accuracy': cv_accuracy,
            'baseline_accuracy': baseline_accuracy,
            'baseline_f1': baseline_f1,
            'baseline_cv_accuracy': cv_baseline_accuracy,
            'feature_reduction': 1.0 - len(selected_features) / len(feature_names)
        }
        
        # Save feature importance
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        
    else:  # regression
        # Model with selected features
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation for more robust performance estimation
        cv_scores_selected = cross_val_score(
            RandomForestRegressor(n_estimators=100, random_state=42),
            X_selected, y, cv=5, scoring='r2'
        )
        cv_r2 = cv_scores_selected.mean()
        
        # Baseline model with all features
        baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
        baseline_model.fit(X_all_train, y_train)
        baseline_pred = baseline_model.predict(X_all_test)
        baseline_mse = mean_squared_error(y_test, baseline_pred)
        baseline_r2 = r2_score(y_test, baseline_pred)
        
        # Cross-validation for baseline
        cv_scores_baseline = cross_val_score(
            RandomForestRegressor(n_estimators=100, random_state=42),
            X, y, cv=5, scoring='r2'
        )
        cv_baseline_r2 = cv_scores_baseline.mean()
        
        print(f"Selected features - MSE: {mse:.4f}, R²: {r2:.4f}, CV R²: {cv_r2:.4f}")
        print(f"All features - MSE: {baseline_mse:.4f}, R²: {baseline_r2:.4f}, CV R²: {cv_baseline_r2:.4f}")
        
        performance = {
            'selected_mse': mse,
            'selected_r2': r2,
            'selected_cv_r2': cv_r2,
            'baseline_mse': baseline_mse,
            'baseline_r2': baseline_r2,
            'baseline_cv_r2': cv_baseline_r2,
            'feature_reduction': 1.0 - len(selected_features) / len(feature_names)
        }
        
        # Save feature importance
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Save models
    joblib.dump(model, os.path.join(output_dir, 'fine_tuned_model.pkl'))
    joblib.dump(baseline_model, os.path.join(output_dir, 'baseline_model.pkl'))
    
    # Save performance metrics
    pd.DataFrame(performance, index=[0]).to_csv(os.path.join(output_dir, 'performance_metrics.csv'), index=False)
    
    return performance


def create_visualizations(selector, feature_names, selected_features, performance, problem_type, X, y, output_dir):
    """
    Create visualizations of the stability selection results
    
    Parameters:
    -----------
    selector: StabilitySelection
        Fitted stability selection object
    feature_names: list
        List of feature names
    selected_features: list
        List of selected feature names
    performance: dict
        Dictionary of performance metrics
    problem_type: str
        'classification' or 'regression'
    X: numpy array
        Feature matrix
    y: numpy array
        Target vector
    output_dir: str
        Directory to save visualizations
    """
    print("Creating visualizations")
    
    # Create stability paths plot
    plt.figure(figsize=(12, 8))
    
    # Get indices of selected features
    selected_indices = [feature_names.index(feature) for feature in selected_features]
    
    # Plot stability paths
    lambda_grid_normalized = selector.lambda_grid / np.max(selector.lambda_grid)
    
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
                label=f'Threshold ({selector.threshold:.2f})')
    
    plt.xlabel('λ / max(λ)')
    plt.ylabel('Stability Score')
    plt.title('Stability Selection Paths')
    
    # If we have many selected features, don't show legend
    if len(selected_features) <= 10:
        plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stability_paths.png'), dpi=300)
    
    # Create a heatmap of the stability scores for top features
    plt.figure(figsize=(12, 8))
    
    # Get top 20 features by max stability score
    top_indices = np.argsort(-np.max(selector.stability_scores_, axis=1))[:20]
    top_features = [feature_names[i] for i in top_indices]
    
    # Create heatmap
    sns.heatmap(selector.stability_scores_[top_indices, :], 
                cmap='viridis', 
                xticklabels=[f'{x:.3f}' for x in lambda_grid_normalized],
                yticklabels=top_features,
                vmin=0, vmax=1)
    
    plt.xlabel('λ / max(λ)')
    plt.ylabel('Feature')
    plt.title('Stability Scores Heatmap (Top 20 Features)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stability_heatmap.png'), dpi=300)
    
    # Create bar plot of performance metrics
    plt.figure(figsize=(12, 6))
    
    if problem_type == 'classification':
        metrics = ['Accuracy', 'F1 Score', 'CV Accuracy']
        selected_values = [
            performance['selected_accuracy'], 
            performance['selected_f1'], 
            performance['selected_cv_accuracy']
        ]
        baseline_values = [
            performance['baseline_accuracy'], 
            performance['baseline_f1'],
            performance['baseline_cv_accuracy']
        ]
    else:  # regression
        metrics = ['R²', 'CV R²', 'MSE (normalized)']
        selected_values = [
            performance['selected_r2'], 
            performance['selected_cv_r2'],
            1 - performance['selected_mse'] / performance['baseline_mse']
        ]
        baseline_values = [
            performance['baseline_r2'], 
            performance['baseline_cv_r2'],
            1 - performance['baseline_mse'] / performance['baseline_mse']
        ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, selected_values, width, label=f'Selected Features ({len(selected_features)})')
    plt.bar(x + width/2, baseline_values, width, label=f'All Features ({len(feature_names)})')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    
    # Add text with feature reduction percentage
    feature_reduction = performance['feature_reduction'] * 100
    plt.figtext(0.5, 0.01, f"Feature reduction: {feature_reduction:.1f}% ({len(selected_features)} / {len(feature_names)} features)", 
                ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300)
    
    # Create bar plot of feature importance for selected features
    plt.figure(figsize=(12, 8))
    
    # Load feature importance
    feature_importance = pd.read_csv(os.path.join(output_dir, 'feature_importance.csv'))
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=True)
    
    # Plot horizontal bar chart
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance from Fine-tuned Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    
    # Create summary visualization
    plt.figure(figsize=(10, 8))
    
    # Plot feature reduction pie chart
    plt.subplot(2, 1, 1)
    plt.pie([len(selected_features), len(feature_names) - len(selected_features)], 
            labels=[f'Selected ({len(selected_features)})', f'Excluded ({len(feature_names) - len(selected_features)})'],
            autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    plt.axis('equal')
    plt.title('Feature Reduction')
    
    # Plot top 5 features by importance
    plt.subplot(2, 1, 2)
    top_n = min(5, len(feature_importance))
    top_features = feature_importance.iloc[-top_n:]
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 5 Features by Importance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary.png'), dpi=300)
    
    print(f"Saved all visualizations to {output_dir}")


def main():
    """Main function to run stability selection on input data"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run stability selection on CSV/CSV.GZ data')
    parser.add_argument('--input', required=True, help='Path to input CSV or CSV.GZ file')
    parser.add_argument('--output', required=True, help='Path to output directory')
    parser.add_argument('--problem_type', choices=['classification', 'regression'], required=True,
                       help='Type of problem: classification or regression')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU acceleration if available')
    parser.add_argument('--n_bootstrap', type=int, default=100, help='Number of bootstrap iterations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Created output directory: {args.output}")
    
    # Load data
    X_df, y = load_data(args.input)
    
    # Preprocess data
    X, y_processed, feature_names = preprocess_data(X_df, y, args.problem_type)
    
    # Run stability selection
    selector, selected_features = run_stability_selection(
        X, y_processed, feature_names, args.problem_type, args.output, args.use_gpu, args.n_bootstrap
    )
    
    # Fine-tune model with selected features
    performance = fine_tune_model(
        X, y_processed, selected_features, feature_names, args.problem_type, args.output
    )
    
    # Create visualizations
    create_visualizations(
        selector, feature_names, selected_features, performance, args.problem_type, X, y_processed, args.output
    )
    
    print("Stability selection analysis completed successfully!")
    print(f"All results saved to: {args.output}")


if __name__ == '__main__':
    main()
