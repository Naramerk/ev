import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def plot_convergence(output_dir, logbook, meta_feature, mutation_type, meta_values, trial=0):
    """
    Plot convergence of the genetic algorithm for vector-valued meta-features.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save plot
    logbook : deap.tools.Logbook
        Logbook from genetic algorithm
    meta_feature : str
        Meta-feature being optimized
    mutation_type : str
        Type of mutation used
    meta_values : dict
        Dictionary with initial, target, and final meta-feature values as vectors
    trial : int
        Trial number
        
    Returns:
    --------
    str
        Path to the saved plot
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Extract data from logbook
    generations = logbook.select("gen")
    min_fitness = logbook.select("min")
    avg_fitness = logbook.select("avg")
    if "std" in logbook.header:
        std_fitness = logbook.select("std")
    else:
        std_fitness = [0] * len(min_fitness)
    
    # Create figure for fitness convergence
    plt.figure(figsize=(10, 6))
    
    # Plot fitness values
    plt.plot(generations, min_fitness, 'r-', label="Best Fitness", linewidth=2)
    plt.plot(generations, avg_fitness, 'b--', label="Average Fitness", linewidth=2)
    
    # Add error bands for standard deviation
    if np.sum(std_fitness) > 0:
        avg_array = np.array(avg_fitness)
        std_array = np.array(std_fitness)
        plt.fill_between(generations, 
                        avg_array - std_array, 
                        avg_array + std_array, 
                        alpha=0.2, color='b')
    
    # Add horizontal line for perfect match (fitness = 0)
    plt.axhline(y=0, color='g', linestyle=':', label="Perfect Match")
    
    # Customize plot
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Fitness (Euclidean Distance)", fontsize=12)
    plt.title(f"Convergence for {meta_feature} with {mutation_type} Mutation", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    # Calculate Euclidean distance between vectors
    initial_vec = np.array(meta_values['initial'])
    target_vec = np.array(meta_values['target'])
    final_vec = np.array(meta_values['final'])
    
    # Calculate distances
    init_to_target_dist = euclidean(initial_vec, target_vec)
    final_to_target_dist = euclidean(final_vec, target_vec)
    improvement = init_to_target_dist - final_to_target_dist
    
    # Convert vector lengths to strings for display
    init_len = f"({len(initial_vec)} elements)"
    target_len = f"({len(target_vec)} elements)"
    final_len = f"({len(final_vec)} elements)"
    
    # Add text box with vector information
    textstr = '\n'.join([
        f"Meta-feature: {meta_feature}",
        f"Initial vector: {init_len}",
        f"Target vector: {target_len}",
        f"Final vector: {final_len}",
        f"Initial distance: {init_to_target_dist:.4f}",
        f"Final distance: {final_to_target_dist:.4f}",
        f"Improvement: {improvement:.4f}"
    ])
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top', bbox=props)
    
    # Save the figure
    filepath = os.path.join(output_dir, f"convergence_{meta_feature}_{mutation_type}_trial{trial}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    
    # Add vector visualization if vectors are not too large (<=10 dimensions)
    if len(initial_vec) <= 10 and len(target_vec) <= 10 and len(final_vec) <= 10:
        # Create second figure for vector comparison
        plt.figure(figsize=(12, 6))
        
        # Prepare indices for x-axis
        x = np.arange(len(initial_vec))
        width = 0.25
        
        # Create bar chart comparing vectors
        plt.bar(x - width, initial_vec, width, label='Initial', color='blue', alpha=0.7)
        plt.bar(x, target_vec, width, label='Target', color='green', alpha=0.7)
        plt.bar(x + width, final_vec, width, label='Final', color='red', alpha=0.7)
        
        # Add labels and title
        plt.xlabel('Vector Component', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title(f'Vector Comparison for {meta_feature}', fontsize=14)
        plt.xticks(x, [f'Dim-{i+1}' for i in range(len(initial_vec))])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        vector_filepath = os.path.join(output_dir, f"vector_comparison_{meta_feature}_{mutation_type}_trial{trial}.png")
        plt.savefig(vector_filepath, dpi=150)
        plt.close()
        
        # If vectors are too large (>10 dimensions), create a PCA visualization
    else:
        try:
            # Stack vectors for PCA
            stacked_vectors = np.vstack([initial_vec, target_vec, final_vec])
            
            # Apply PCA to reduce to 2 dimensions
            pca = PCA(n_components=2)
            reduced_vectors = pca.fit_transform(stacked_vectors)
            
            # Create a scatter plot
            plt.figure(figsize=(8, 8))
            plt.scatter(reduced_vectors[0, 0], reduced_vectors[0, 1], color='blue', s=100, label='Initial')
            plt.scatter(reduced_vectors[1, 0], reduced_vectors[1, 1], color='green', s=100, label='Target')
            plt.scatter(reduced_vectors[2, 0], reduced_vectors[2, 1], color='red', s=100, label='Final')
            
            # Add arrows to show direction of evolution
            plt.arrow(reduced_vectors[0, 0], reduced_vectors[0, 1], 
                      reduced_vectors[2, 0] - reduced_vectors[0, 0], 
                      reduced_vectors[2, 1] - reduced_vectors[0, 1], 
                      color='black', width=0.01, head_width=0.05)
            
            # Add labels
            plt.title(f"PCA visualization of {meta_feature} vectors", fontsize=14)
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)", fontsize=12)
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)", fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True)
            
            # Save the figure
            pca_filepath = os.path.join(output_dir, f"pca_vectors_{meta_feature}_{mutation_type}_trial{trial}.png")
            plt.savefig(pca_filepath, dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error creating PCA visualization: {e}")
    
    return filepath

def plot_data_comparison(output_dir, source_data, target_data, synthetic_data, meta_feature, mutation_type, shift_type, trial=0):
    """
    Create pairplots to visualize and compare source, target, and synthetic data.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save plots
    source_data : DataFrame
        Source domain data
    target_data : DataFrame
        Target domain data
    synthetic_data : DataFrame
        Generated synthetic data
    meta_feature : str
        Meta-feature being optimized
    mutation_type : str
        Type of mutation used
    shift_type : str
        Type of distribution shift
    trial : int
        Trial number
        
    Returns:
    --------
    str
        Path to the saved plot
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert to DataFrames if needed
    if not isinstance(source_data, pd.DataFrame):
        source_data = pd.DataFrame(source_data)
    if not isinstance(target_data, pd.DataFrame):
        target_data = pd.DataFrame(target_data)
    if not isinstance(synthetic_data, pd.DataFrame):
        synthetic_data = pd.DataFrame(synthetic_data)
    
    # Rename columns if no names
    if all(source_data.columns.str.startswith('0')) or all(source_data.columns.str.match(r'\d+')):
        cols = [f'x{i+1}' for i in range(source_data.shape[1]-1)] + ['y']
        source_data.columns = cols
        target_data.columns = cols
        synthetic_data.columns = cols
    
    # Prepare data for pairplot
    source_sample = source_data.copy()
    target_sample = target_data.copy()
    synthetic_sample = synthetic_data.copy()
    
    # Limit number of samples for pairplot (for efficiency)
    max_samples = 500
    if len(source_sample) > max_samples:
        source_sample = source_sample.sample(max_samples, random_state=42)
    if len(target_sample) > max_samples:
        target_sample = target_sample.sample(max_samples, random_state=42)
    if len(synthetic_sample) > max_samples:
        synthetic_sample = synthetic_sample.sample(max_samples, random_state=42)
    
    # Add domain label
    source_sample['domain'] = 'Source'
    target_sample['domain'] = 'Target'
    synthetic_sample['domain'] = 'Synthetic'
    
    # Combine data
    combined_data = pd.concat([source_sample, target_sample, synthetic_sample])
    
    # Create pairplot
    plt.figure(figsize=(15, 15))
    g = sns.pairplot(
        combined_data, 
        hue='domain',
        diag_kind='kde',
        plot_kws={'alpha': 0.5, 's': 15},
        height=2.5,
        aspect=1
    )
    
    # Customize plot
    plt.suptitle(
        f"Data Comparison\nMeta-feature: {meta_feature}, Mutation: {mutation_type}, Shift: {shift_type}",
        y=1.02,
        fontsize=16
    )
    
    # Save plot
    filepath = os.path.join(output_dir, f"pairplot_{meta_feature}_{mutation_type}_{shift_type}_trial{trial}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create distribution plots for each feature
    features = source_data.columns.tolist()[:-1]  # Exclude 'y'
    n_features = len(features)
    
    # Determine grid size
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Extract data ranges for all distributions to set a consistent x-range
        all_data = np.concatenate([
            source_sample[feature].values,
            target_sample[feature].values,
            synthetic_sample[feature].values
        ])
        data_min = np.min(all_data)
        data_max = np.max(all_data)
        
        # Add 10% padding on both sides
        padding = (data_max - data_min) * 0.1
        x_min = data_min - padding
        x_max = data_max + padding
        
        # Create a common x-axis range for plotting
        x_range = np.linspace(x_min, x_max, 1000)
        
        # Plot source distribution with higher z-order to ensure visibility
        sns.kdeplot(source_sample[feature], ax=ax, label='Source', color='blue', zorder=10)
        sns.kdeplot(target_sample[feature], ax=ax, label='Target', color='red', zorder=5)
        sns.kdeplot(synthetic_sample[feature], ax=ax, label='Synthetic', color='green', zorder=1)
        
        # Set consistent x-axis limits
        ax.set_xlim(x_min, x_max)
        
        # Increase linewidth for better visibility
        for line in ax.get_lines():
            line.set_linewidth(2)
            
        ax.set_title(f"Distribution of {feature}")
        ax.legend()
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(
        f"Feature Distributions\nMeta-feature: {meta_feature}, Mutation: {mutation_type}, Shift: {shift_type}",
        y=1.02,
        fontsize=16
    )
    plt.tight_layout()
    
    # Save plot
    filepath = os.path.join(output_dir, f"distributions_{meta_feature}_{mutation_type}_{shift_type}_trial{trial}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    
    Parameters
    ----------
    x, y : array-like
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from the square root of the variance
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    
    # calculating the standard deviation of y from the square root of the variance
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_meta_feature_variability(output_dir, dataset, meta_feature, n_samples=10, sample_size_range=(0.1, 0.9)):
    """
    Analyze and plot the variability of a vector-valued meta-feature across different subsets of a dataset.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save plots
    dataset : DataFrame
        Dataset to analyze
    meta_feature : str
        Meta-feature to analyze
    n_samples : int
        Number of subsamples to create
    sample_size_range : tuple
        Range of sample sizes as fraction of dataset (min, max)
    
    Returns:
    --------
    tuple
        (values, filepath) - meta-feature values and plot filepath
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Convert to DataFrame if needed
    if not isinstance(dataset, pd.DataFrame):
        dataset = pd.DataFrame(dataset)
    
    # Compute meta-feature on full dataset
    from pymfe.mfe import MFE
    
    def compute_meta_feature(data, meta_feature):
        """Helper to compute the vector-valued meta-feature"""
        if isinstance(data, pd.DataFrame):
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
        else:
            X = data[:, :-1]
            y = data[:, -1]
        
        # Replace NaN values
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        
        mfe = MFE(features=[meta_feature], summary=None)
        mfe.fit(X, y)
        ft = mfe.extract()[1][0]
        return ft
    
    full_value = compute_meta_feature(dataset, meta_feature)
    
    # Generate random subsets of different sizes
    values = []
    sample_sizes = []
    
    min_size, max_size = sample_size_range
    for _ in range(n_samples):
        # Generate random sample size
        size_fraction = np.random.uniform(min_size, max_size)
        sample_size = int(len(dataset) * size_fraction)
        
        # Sample the dataset
        sample = dataset.sample(sample_size, replace=False)
        
        # Compute meta-feature
        value = compute_meta_feature(sample, meta_feature)
        
        # Store results
        if not np.isnan(value).any():
            values.append(value)
            sample_sizes.append(sample_size)
    
    # If no valid values were computed, return early
    if len(values) == 0:
        print(f"No valid meta-feature values could be computed for {meta_feature}")
        return {
            'values': [],
            'sample_sizes': [],
            'full_value': full_value,
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'range': None,
            'cov': None
        }, None
    
    # Convert to numpy arrays for easier manipulation
    values_array = np.array(values)
    
    # Calculate statistics for vector norms
    norms = np.linalg.norm(values_array, axis=1)
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    min_norm = np.min(norms)
    max_norm = np.max(norms)
    range_norm = max_norm - min_norm
    
    # Calculate element-wise statistics
    mean_vec = np.mean(values_array, axis=0)
    std_vec = np.std(values_array, axis=0)
    min_vec = np.min(values_array, axis=0)
    max_vec = np.max(values_array, axis=0)
    range_vec = max_vec - min_vec
    
    # Create plots based on dimensionality of meta-feature vectors
    vector_dim = len(full_value)
    
    if vector_dim <= 10:  # For small dimensionality, we can visualize directly
        # Create figure for vector elements
        plt.figure(figsize=(12, 6))
        
        # Plot each vector dimension
        for i in range(vector_dim):
            plt.plot(sample_sizes, values_array[:, i], 'o-', alpha=0.7, label=f'Dim-{i+1}')
            
        # Add horizontal lines for full dataset value components
        for i in range(vector_dim):
            plt.axhline(y=full_value[i], color=f'C{i}', linestyle='--', alpha=0.7)
            
        # Customize plot
        plt.xlabel("Sample Size", fontsize=12)
        plt.ylabel(f"Meta-feature: {meta_feature} Values", fontsize=12)
        plt.title(f"Variability of {meta_feature} Vector Components with Different Sample Sizes", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='best')
        
        # Save the figure
        components_filepath = os.path.join(output_dir, f"variability_components_{meta_feature}.png")
        plt.savefig(components_filepath, dpi=150)
        plt.close()
        
        # Create figure for vector norms
        plt.figure(figsize=(10, 6))
        
        # Plot norms vs sample sizes
        plt.scatter(sample_sizes, norms, alpha=0.7)
        
        # Add horizontal line for full dataset norm
        full_norm = np.linalg.norm(full_value)
        plt.axhline(y=full_norm, color='r', linestyle='--', 
                    label=f'Full dataset ({len(dataset)} samples): {full_norm:.4f}')
        
        # Fit a trend line
        if len(norms) > 1:
            try:
                z = np.polyfit(sample_sizes, norms, 1)
                p = np.poly1d(z)
                plt.plot(sorted(sample_sizes), p(sorted(sample_sizes)), 
                        "r--", alpha=0.5, label=f"Trend: y={z[0]:.2e}x+{z[1]:.2f}")
            except:
                pass
        
        # Customize plot
        plt.xlabel("Sample Size", fontsize=12)
        plt.ylabel(f"Norm of {meta_feature} Vector", fontsize=12)
        plt.title(f"Variability of {meta_feature} Vector Norm with Different Sample Sizes", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Add text box with statistics on norms
        textstr = '\n'.join([
            f"Statistics for {meta_feature} Vector Norms:",
            f"Mean: {mean_norm:.4f}",
            f"Std Dev: {std_norm:.4f}",
            f"Range: {range_norm:.4f}",
            f"Min: {min_norm:.4f}",
            f"Max: {max_norm:.4f}",
            f"CoV: {(std_norm / mean_norm if mean_norm != 0 else 0):.4f}"
        ])
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=props)
        
        # Save the figure
        norm_filepath = os.path.join(output_dir, f"variability_norm_{meta_feature}.png")
        plt.savefig(norm_filepath, dpi=150)
        plt.close()
        
    else:  # For high-dimensional vectors, use PCA for visualization
        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=2)
        try:
            # Stack full value with sample values
            all_values = np.vstack([full_value, values_array])
            reduced_values = pca.fit_transform(all_values)
            
            # Extract reduced coordinates
            full_reduced = reduced_values[0]
            samples_reduced = reduced_values[1:]
            
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            
            # Plot samples with color based on sample size
            sc = plt.scatter(samples_reduced[:, 0], samples_reduced[:, 1], 
                         c=sample_sizes, cmap='viridis', alpha=0.7, s=100)
            
            # Plot full dataset value
            plt.scatter(full_reduced[0], full_reduced[1], 
                      color='red', s=200, marker='*', label='Full Dataset')
            
            # Add confidence ellipse
            confidence_ellipse(samples_reduced[:, 0], samples_reduced[:, 1], 
                             plt.gca(), n_std=2, edgecolor='black', linestyle='--')
            
            # Add colorbar
            cbar = plt.colorbar(sc)
            cbar.set_label('Sample Size', fontsize=12)
            
            # Customize plot
            plt.title(f"PCA of {meta_feature} Vectors from Different Subsets", fontsize=14)
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)", fontsize=12)
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)", fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True)
            
            # Save the figure
            pca_filepath = os.path.join(output_dir, f"variability_pca_{meta_feature}.png")
            plt.savefig(pca_filepath, dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"Error creating PCA visualization: {e}")
    
    # Create main variability plot (scalar summary)
    filepath = os.path.join(output_dir, f"variability_{meta_feature}.png")
    
    # Return meta-feature values and filepath
    return {
        'values': values,
        'sample_sizes': sample_sizes,
        'full_value': full_value,
        'mean_norm': mean_norm,
        'std_norm': std_norm,
        'min_norm': min_norm,
        'max_norm': max_norm,
        'range_norm': range_norm,
        'cov_norm': std_norm / mean_norm if mean_norm != 0 else None,
        'mean_vec': mean_vec.tolist() if hasattr(mean_vec, 'tolist') else mean_vec,
        'std_vec': std_vec.tolist() if hasattr(std_vec, 'tolist') else std_vec,
        'vector_dim': vector_dim
    }, filepath 