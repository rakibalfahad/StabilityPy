"""
Synthetic Data Generator for Stability Selection

This script generates synthetic data for testing stability selection.
It can create both classification and regression datasets with controlled
sparsity and noise levels.

Usage:
    python synthetic_data_generator.py --output data.csv --problem_type classification
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.utils import check_random_state


def generate_classification_data(n_samples=1000, n_features=100, n_informative=10,
                                n_redundant=10, n_repeated=0, n_classes=2,
                                random_state=42, add_noise=0.1, compress=False):
    """
    Generate synthetic classification data
    
    Parameters:
    -----------
    n_samples: int
        Number of samples
    n_features: int
        Number of features
    n_informative: int
        Number of informative features
    n_redundant: int
        Number of redundant features
    n_repeated: int
        Number of repeated features
    n_classes: int
        Number of classes
    random_state: int
        Random state
    add_noise: float
        Noise level to add (0 to 1)
    compress: bool
        Whether to compress the output file
        
    Returns:
    --------
    X: numpy array
        Feature matrix
    y: numpy array
        Target vector
    feature_indices: dict
        Dictionary mapping feature types to their indices
    """
    print(f"Generating classification data with {n_samples} samples and {n_features} features")
    print(f"Informative features: {n_informative}, Redundant features: {n_redundant}, Repeated features: {n_repeated}")
    
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
        print(f"Added noise with level {add_noise}")
    
    # Track feature types
    feature_indices = {
        'informative': list(range(n_informative)),
        'redundant': list(range(n_informative, n_informative + n_redundant)),
        'repeated': list(range(n_informative + n_redundant, n_informative + n_redundant + n_repeated)),
        'random': list(range(n_informative + n_redundant + n_repeated, n_features))
    }
    
    return X, y, feature_indices


def generate_regression_data(n_samples=1000, n_features=100, n_informative=10,
                           effective_rank=None, tail_strength=0.5, noise=0.1,
                           random_state=42, add_noise=0.1, compress=False):
    """
    Generate synthetic regression data
    
    Parameters:
    -----------
    n_samples: int
        Number of samples
    n_features: int
        Number of features
    n_informative: int
        Number of informative features
    effective_rank: int or None
        Effective rank of the data matrix
    tail_strength: float
        Relative importance of the fat noisy tail of the singular values
    noise: float
        Standard deviation of the gaussian noise
    random_state: int
        Random state
    add_noise: float
        Additional noise level to add (0 to 1)
    compress: bool
        Whether to compress the output file
        
    Returns:
    --------
    X: numpy array
        Feature matrix
    y: numpy array
        Target vector
    feature_indices: dict
        Dictionary mapping feature types to their indices
    """
    print(f"Generating regression data with {n_samples} samples and {n_features} features")
    print(f"Informative features: {n_informative}, Noise level: {noise}")
    
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
        print(f"Added noise with level {add_noise}")
    
    # Track informative features
    informative_indices = np.where(coef != 0)[0]
    
    feature_indices = {
        'informative': informative_indices.tolist(),
        'non_informative': [i for i in range(n_features) if i not in informative_indices]
    }
    
    return X, y, feature_indices


def save_data(X, y, feature_indices, output_path, compress=False, add_feature_names=True, problem_type=None):
    """
    Save generated data to file
    
    Parameters:
    -----------
    X: numpy array
        Feature matrix
    y: numpy array
        Target vector
    feature_indices: dict
        Dictionary mapping feature types to their indices
    output_path: str
        Path to save the data
    compress: bool
        Whether to compress the output file
    add_feature_names: bool
        Whether to add feature names
    problem_type: str
        'classification' or 'regression'
    """
    # Create DataFrame
    if add_feature_names:
        feature_names = []
        for i in range(X.shape[1]):
            feature_type = 'unknown'
            for ftype, indices in feature_indices.items():
                if i in indices:
                    feature_type = ftype
                    break
            feature_names.append(f'{feature_type}_feature_{i}')
        
        df = pd.DataFrame(X, columns=feature_names)
    else:
        df = pd.DataFrame(X)
    
    # Add target
    if problem_type == 'classification':
        df['class'] = y
    else:
        df['target'] = y
    
    # Save to file
    if compress or output_path.endswith('.gz'):
        if not output_path.endswith('.gz'):
            output_path += '.gz'
        df.to_csv(output_path, index=False, compression='gzip')
        print(f"Data saved to {output_path} (compressed)")
    else:
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
    
    # Save feature indices for reference
    indices_path = os.path.splitext(output_path)[0] + '_feature_indices.csv'
    
    indices_df = pd.DataFrame(columns=['feature_index', 'feature_type'])
    
    for feature_type, indices in feature_indices.items():
        for idx in indices:
            indices_df = indices_df._append({
                'feature_index': idx,
                'feature_type': feature_type
            }, ignore_index=True)
    
    indices_df.to_csv(indices_path, index=False)
    print(f"Feature indices saved to {indices_path}")


def main():
    """Main function to generate synthetic data"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic data for stability selection')
    parser.add_argument('--output', required=True, help='Path to output CSV or CSV.GZ file')
    parser.add_argument('--problem_type', choices=['classification', 'regression'], required=True,
                       help='Type of problem: classification or regression')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--n_features', type=int, default=100, help='Number of features')
    parser.add_argument('--n_informative', type=int, default=10, help='Number of informative features')
    parser.add_argument('--noise', type=float, default=0.1, help='Noise level')
    parser.add_argument('--compress', action='store_true', help='Compress output file')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--no_feature_names', action='store_true', help='Do not add feature names')
    
    args = parser.parse_args()
    
    # Generate data
    if args.problem_type == 'classification':
        X, y, feature_indices = generate_classification_data(
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_informative=args.n_informative,
            random_state=args.random_state,
            add_noise=args.noise,
            compress=args.compress
        )
    else:  # regression
        X, y, feature_indices = generate_regression_data(
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_informative=args.n_informative,
            random_state=args.random_state,
            add_noise=args.noise,
            compress=args.compress
        )
    
    # Save data
    save_data(
        X, y, feature_indices, args.output, 
        compress=args.compress, 
        add_feature_names=not args.no_feature_names,
        problem_type=args.problem_type
    )
    
    print("Data generation completed successfully!")


if __name__ == '__main__':
    main()
