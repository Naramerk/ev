import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pymfe.mfe import MFE
from deap import base, creator, tools, algorithms
from ForestDiffusion import ForestDiffusionModel
import random
import warnings
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from functools import partial
import json
from multiprocessing import Pool
from bamt.networks import ContinuousBN
import glob
from sklearn.preprocessing import LabelEncoder


plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings("ignore")

# Clear any existing DEAP creator classes to avoid conflicts
if hasattr(creator, "FitnessMin"):
    del creator.FitnessMin
if hasattr(creator, "Individual"):
    del creator.Individual

# Create DEAP classes at the module level
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Global variables for storing meta-feature values
init_val = 'a'
fin_val = 'a'
tar_val = 'a'
fit_value = float('inf')

def generate_shifted_classification_data(n_samples=500, seed=42, shift_type='all'):
    """
    Generates source and target classification data with multiple distribution shifts.
    
    Features:
    - x1: Covariate shift (mean change)
    - x2: Variance shift
    - x3, x4: Correlation shift
    - x5: Concept shift (feature importance change)
    
    Labels:
    - Concept shift (changed decision boundary)
    - Prior probability shift (class balance change)
    
    Args:
        n_samples (int): Number of samples per domain
        seed (int): Random seed
        
    Returns:
        source_df, target_df: DataFrames with features and labels
    """
    np.random.seed(seed)
    
    # Generate base features using sklearn's make_classification
    X, y = make_classification(
        n_samples=2*n_samples,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        flip_y=0.05,
        random_state=seed
    )
    
    # Split into source and target base data
    source_X = X[:n_samples]
    source_y = y[:n_samples]
    target_X = X[n_samples:]
    target_y = y[n_samples:]
    
    # Create DataFrame structures
    source_df = pd.DataFrame(source_X, columns=[f'x{i+1}' for i in range(5)])
    source_df['y'] = source_y
    target_df = pd.DataFrame(target_X, columns=[f'x{i+1}' for i in range(5)])
    target_df['y'] = target_y

    # Apply distribution shifts to target domain
    # -------------------------------------------------
    
    # 1. Covariate Shift (x1: mean shift)
    target_df['x1'] = target_df['x1'] + 3.0
    
    # 2. Variance Shift (x2: increased variance)
    target_df['x2'] = target_df['x2'] * 2.5
    
    # 3. Correlation Shift (x3 & x4: break correlation)
    target_df['x3'] = np.random.normal(0, 1, n_samples)
    target_df['x4'] = np.random.normal(0, 1, n_samples)
    
    # 4. Concept Shift (x5: change relationship with label)
    # Modify existing labels based on new relationship
    target_df['y'] = np.where(
        (target_df['x5'] > 0.2) & (target_df['x3'] < 0.8),
        1 - target_df['y'],  # Flip labels for concept shift
        target_df['y']
    )
    
    class_0 = target_df[target_df.y == 0]
    class_1 = target_df[target_df.y == 1]

    # Calculate desired samples for each class
    n_class_0 = int(n_samples * 0.8)
    n_class_1 = int(n_samples * 0.2)

    # Sample with replacement if needed
    sampled_class_0 = class_0.sample(n=n_class_0, 
                                   replace=len(class_0) < n_class_0, 
                                   random_state=seed)
    sampled_class_1 = class_1.sample(n=n_class_1, 
                                   replace=len(class_1) < n_class_1, 
                                   random_state=seed)

    target_df = pd.concat([sampled_class_0, sampled_class_1]) \
                .sample(frac=1, random_state=seed)  # Shuffle

    return source_df, target_df

def generate_isolated_shift_data(shift_type, n_samples=500, seed=42):
    """
    Generate synthetic data with a single specified distribution shift type.
    
    Parameters:
    -----------
    shift_type : str
        Type of shift to apply:
        - 'covariate' - Covariate shift (feature distribution change)
        - 'concept' - Concept shift (relationship between features and target changes)
        - 'prior' - Prior probability shift (class balance changes)
        - 'variance' - Variance shift (feature variance changes)
        - 'correlation' - Correlation shift (correlation between features changes)
    n_samples : int
        Number of samples per domain
    seed : int
        Random seed
        
    Returns:
    --------
    source_df, target_df : DataFrame
        DataFrames for source and target domains
    """
    np.random.seed(seed)
    
    # Generate base features
    X, y = make_classification(
        n_samples=2*n_samples,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        flip_y=0.05,
        random_state=seed
    )
    
    # Split into source and target base data
    source_X = X[:n_samples]
    source_y = y[:n_samples]
    target_X = X[n_samples:]
    target_y = y[n_samples:]
    
    # Create DataFrame structures
    source_df = pd.DataFrame(source_X, columns=[f'x{i+1}' for i in range(5)])
    source_df['y'] = source_y
    target_df = pd.DataFrame(target_X, columns=[f'x{i+1}' for i in range(5)])
    target_df['y'] = target_y
    
    # Apply the specific distribution shift
    if shift_type == 'covariate':
        # Covariate Shift - change mean of features
        target_df['x1'] = target_df['x1'] + 3.0
        target_df['x2'] = target_df['x2'] - 2.0
        
    elif shift_type == 'concept':
        # Concept Shift - change relationship between features and target
        target_df['y'] = np.where(
            (target_df['x5'] > 0.2), 
            1 - target_df['y'],  # Flip labels for concept shift
            target_df['y']
        )
        
    elif shift_type == 'prior':
        # Prior Probability Shift - change class balance
        class_0 = target_df[target_df.y == 0]
        class_1 = target_df[target_df.y == 1]

        # Calculate desired samples for each class (80/20 split)
        n_class_0 = int(n_samples * 0.8)
        n_class_1 = int(n_samples * 0.2)

        # Sample with replacement if needed
        sampled_class_0 = class_0.sample(n=n_class_0, 
                                       replace=len(class_0) < n_class_0, 
                                       random_state=seed)
        sampled_class_1 = class_1.sample(n=n_class_1, 
                                       replace=len(class_1) < n_class_1, 
                                       random_state=seed)

        target_df = pd.concat([sampled_class_0, sampled_class_1]) \
                    .sample(frac=1, random_state=seed)  # Shuffle
                    
    elif shift_type == 'variance':
        # Variance Shift - change variance of features
        target_df['x2'] = target_df['x2'] * 2.5
        target_df['x3'] = target_df['x3'] * 0.5
        
    elif shift_type == 'correlation':
        # Correlation Shift - break correlation between features
        target_df['x3'] = np.random.normal(0, 1, n_samples)
        target_df['x4'] = np.random.normal(0, 1, n_samples)
    
    return source_df, target_df 