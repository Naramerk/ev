from vector_plots import *
import numpy as np
from vector_plots import *
from preprocess_fitness_mf import *
from shifts import *
import pandas as pd
import numpy as np
import json
import os


def run_shift_convergence_experiment(
    shift_type, 
    meta_features, 
    mutation_types, 
    n_trials=1,
    output_dir="distribution_shift_results",
    n_samples=500,
    scaling_factors=[1],
    generations=100,
    source_file=None,
    target_file=None
):
    """
    Run experiments comparing convergence of meta-features on datasets from real source and target data.
    
    Parameters:
    -----------
    shift_type : str
        Type of distribution shift to test
    meta_features : list
        List of meta-features to evaluate
    mutation_types : list
        List of mutation types to test
    n_trials : int
        Number of trials per configuration
    output_dir : str
        Base directory for saving results
    n_samples : int
        Number of samples to use from datasets (if larger)
    scaling_factors : list
        List of scaling factors to apply to meta-feature target values
    generations : int
        Number of generations for the evolutionary algorithm
    source_file : str
        Path to source data CSV file
    target_file : str
        Path to target data CSV file
        
    Returns:
    --------
    dict
        Results dictionary
    """
    # Create output directory
    experiment_dir = os.path.join(output_dir, f"shift_{shift_type}")
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    # Load source and target datasets
    print(f"Loading datasets for {shift_type} shift...")
    
    if source_file and target_file:
        # Load from provided file paths
        source_df = pd.read_csv(source_file)
        target_df = pd.read_csv(target_file)
        print(f"Loaded data from {source_file} and {target_file}")
    elif shift_type.startswith('file:'):
        # Extract file base name from shift_type
        file_base = shift_type.split(':')[1]
        source_path = f'source/{file_base}.csv'
        target_path = f'target/{file_base}.csv'
        
        # Load data
        source_df = pd.read_csv(source_path)
        target_df = pd.read_csv(target_path)
        print(f"Loaded data from {source_path} and {target_path}")
    else:
        # For backward compatibility - generate synthetic data
        if shift_type != 'all':
            source_df, target_df = generate_isolated_shift_data(shift_type, n_samples=n_samples)
        else:
            source_df, target_df = generate_shifted_classification_data(n_samples=n_samples)
        print("Used synthetic data generation (legacy mode)")
    
    # Limit to n_samples if needed
    if n_samples and len(source_df) > n_samples:
        source_df = source_df.sample(n_samples, random_state=42)
        print(f"Sampled {n_samples} from source dataset")
    
    if n_samples and len(target_df) > n_samples:
        target_df = target_df.sample(n_samples, random_state=42)
        print(f"Sampled {n_samples} from target dataset")
    
    # Handle any data preprocessing required for the specific datasets
    # Drop NAs
    source_df.dropna(inplace=True)
    target_df.dropna(inplace=True)
    
    # Reset indices
    source_df.reset_index(drop=True, inplace=True)
    target_df.reset_index(drop=True, inplace=True)
    
    # Compute initial meta-feature values
    source_values = {}
    target_values = {}
    
    for meta_feature in meta_features:
        source_values[meta_feature] = compute_meta_feature(source_df, meta_feature)
        target_values[meta_feature] = compute_meta_feature(target_df, meta_feature)
        
    # Prepare summary results
    results = {
        'shift_type': shift_type,
        'source_values': source_values,
        'target_values': target_values,
        'experiments': []
    }
    
    # Save source and target datasets
    source_df.to_csv(os.path.join(experiment_dir, f"source_data.csv"), index=False)
    target_df.to_csv(os.path.join(experiment_dir, f"target_data.csv"), index=False)
    
    # Run experiments
    for meta_feature in meta_features:
        # Create directory for this meta-feature
        meta_feature_dir = os.path.join(experiment_dir, meta_feature)
        if not os.path.exists(meta_feature_dir):
            os.makedirs(meta_feature_dir)
            
        # For each scaling factor
        for scaling_factor in scaling_factors:
            # Use the actual target value
            target_meta_value = target_values[meta_feature]
            scaling_description = "actual target"
            
            print(f"Meta-feature values - Initial: {source_values[meta_feature]}, "
                f"Target: {target_meta_value}")
            
            # Create directory for this scaling factor
            scaling_dir = os.path.join(meta_feature_dir, f"scale_{scaling_factor}")
            if not os.path.exists(scaling_dir):
                os.makedirs(scaling_dir)
            
            # Run experiment for each mutation type
            for mutation_type in mutation_types:
                print(f"\n-- Testing mutation type: {mutation_type} --")
                
                # Create directory for this mutation type
                mutation_dir = os.path.join(scaling_dir, mutation_type)
                if not os.path.exists(mutation_dir):
                    os.makedirs(mutation_dir)
                
                # Run multiple trials
                trial_results = []
                
                for trial in range(n_trials):
                    print(f"Running trial {trial+1}/{n_trials}")
                    
                    # Generate synthetic data
                    try:
                        synthetic_data, logbook, original_data, meta_values = generate_synthetic_data(
                            mutation_type=mutation_type,
                            source_data=source_df,
                            meta_feature=meta_feature,
                            target_meta_value=target_meta_value,
                            population_size=100,
                            generations=generations
                        )
                        print('synthetic_data')
                        # Plot convergence
                        convergence_plot = plot_convergence(
                            mutation_dir, 
                            logbook, 
                            meta_feature, 
                            mutation_type, 
                            meta_values, 
                            trial
                        )
                        print('convergence_plot')
                        
                        # Plot data comparison
                        comparison_plot = plot_data_comparison(
                            mutation_dir,
                            source_df,
                            target_df, 
                            synthetic_data,
                            meta_feature,
                            mutation_type,
                            shift_type,
                            trial
                        )
                        print('comparison_plot')
                        # Save synthetic data
                        if isinstance(synthetic_data, pd.DataFrame):
                            synthetic_data.to_csv(
                                os.path.join(mutation_dir, f"synthetic_data_trial{trial}.csv"),
                                index=False
                            )
                        print('synthetic_data')
                        # Store trial result
                        trial_result = {
                            'trial': trial,
                            'initial_value': meta_values['initial'],
                            'target_value': meta_values['target'],
                            'final_value': meta_values['final'],
                            'absolute_error': np.linalg.norm(np.array(meta_values['target']) - np.array(meta_values['final'])),
                            'relative_error': np.linalg.norm(np.array(meta_values['target']) - np.array(meta_values['final'])) / np.linalg.norm(np.array(meta_values['target'])) if np.linalg.norm(np.array(meta_values['target'])) != 0 else float('inf'),
                            'mean_initial_value': np.mean(meta_values['initial']),
                            'mean_target_value': np.mean(meta_values['target']),
                            'mean_final_value': np.mean(meta_values['final']),
                            'mean_absolute_error': abs(np.mean(meta_values['target']) - np.mean(meta_values['final'])),
                            'mean_relative_error': abs(np.mean(meta_values['target']) - np.mean(meta_values['final'])) / abs(np.mean(meta_values['target'])) if np.mean(meta_values['target']) != 0 else float('inf'),
                            'convergence_plot': convergence_plot,
                            'comparison_plot': comparison_plot
                        }
                        print('trial_result')
                        trial_results.append(trial_result)
                        
                    except Exception as e:
                        print(f"Error in trial {trial}: {e}")
                        continue
                
                # Calculate average results across trials
                if trial_results:
                    experiment_result = {
                            'shift_type': shift_type,
                            'meta_feature': meta_feature,
                            'mutation_type': mutation_type,
                            'initial_value': trial_results[0]['initial_value'],
                            'target_value': trial_results[0]['target_value'],
                            'final_value': np.mean([tr['final_value'] for tr in trial_results]),
                            'absolute_error': np.mean([tr['absolute_error'] for tr in trial_results]),
                            'relative_error': np.mean([tr['relative_error'] for tr in trial_results if tr['relative_error'] != float('inf')]),
                            'mean_initial_value': np.mean([tr['mean_initial_value'] for tr in trial_results]),
                            'mean_target_value': np.mean([tr['mean_target_value'] for tr in trial_results]),
                            'mean_final_value': np.mean([tr['mean_final_value'] for tr in trial_results]),
                            'mean_absolute_error': abs(np.mean([tr['mean_absolute_error'] for tr in trial_results])),
                            'mean_relative_error': abs(np.mean([tr['mean_relative_error'] for tr in trial_results])),
                           
                        }
                    
                    results['experiments'].append(experiment_result)
    
    try:
        with open(os.path.join(experiment_dir, 'results.json'), 'w') as f:
                        # Create a custom serializer to handle different types
            def json_serialize(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (set, frozenset)):
                    return list(obj)
                elif isinstance(obj, complex):
                    return str(obj)
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif callable(obj):
                    return str(obj)
                return str(obj)
            
            # Convert to JSON-serializable structure
            json_results = {}
            for key, value in results.items():
                if key == 'experiments':
                    json_results[key] = []
                    for exp in value:
                        json_exp = {}
                        for k, v in exp.items():
                            try:
                                # Try to convert using our custom serializer
                                if isinstance(v, (np.ndarray, list, tuple)):
                                    json_exp[k] = json_serialize(v)
                                else:
                                    json_exp[k] = v
                            except:
                                # If conversion fails, use string representation
                                json_exp[k] = str(v)
                        json_results[key].append(json_exp)
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2, default=json_serialize)
    except Exception as e:
        print(f"Error saving results to JSON: {e}")
        print("Results will be returned but not saved to file.")
    
    return results

def run_meta_feature_variability_experiment(
    datasets=None,
    meta_features=None,
    output_dir="distribution_shift_results/variability",
    n_subsets=20
):
    """
    Analyze the variability of meta-features across different subsets of real datasets.
    
    Parameters:
    -----------
    datasets : list of dicts
        List of dataset specifications, each with 'name', 'file_path', and optionally 'type' keys
        If None, will use default datasets from source/ directory
    meta_features : list
        List of meta-features to analyze
    output_dir : str
        Directory for saving results
    n_subsets : int
        Number of subsets to generate per dataset
        
    Returns:
    --------
    dict
        Results dictionary
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # If meta_features is not provided, use default meta-features
    if meta_features is None:
        meta_features = ['eigenvalues', 'kurtosis', 'iq_range', 'cor', 'cov']
    
    # Prepare results dictionary
    results = {
        'datasets': [],
        'meta_features': meta_features
    }
    
    # If no datasets provided, use default datasets from source directory
    if datasets is None:
        # Find all csv files in source directory
        try:
            source_files = [f for f in os.listdir('source/') if f.endswith('.csv')]
            datasets = [{'name': f.split('.')[0], 'file_path': f'source/{f}'} for f in source_files]
            print(f"Found {len(datasets)} datasets in source/ directory")
        except Exception as e:
            print(f"Error reading source directory: {e}")
            datasets = []
    
    # For each dataset
    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        file_path = dataset_info['file_path']
        dataset_type = dataset_info.get('type', 'tabular')
        
        print(f"\n=== Analyzing dataset: {dataset_name} ===")
        
        # Create directory for this dataset
        dataset_dir = os.path.join(output_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        # Load the dataset
        try:
            # Load data from CSV file
            dataset_df = pd.read_csv(file_path)
            print(f"Loaded dataset from {file_path}: {dataset_df.shape[0]} rows, {dataset_df.shape[1]} columns")
            
            # Data preprocessing
            # Drop NAs
            dataset_df.dropna(inplace=True)
            
            # Reset indices
            dataset_df.reset_index(drop=True, inplace=True)
            
            # Handle specific dataset types or preprocessing
            if dataset_type == 'binary_classification':
                # Ensure target is binary 0/1
                if 'class' in dataset_df.columns:
                    dataset_df['class'] = dataset_df['class'].map({0.0: 0, 1.0: 1}, na_action='ignore')
                elif 'target' in dataset_df.columns:
                    dataset_df['target'] = dataset_df['target'].map({0.0: 0, 1.0: 1}, na_action='ignore')
                elif 'y' in dataset_df.columns:
                    dataset_df['y'] = dataset_df['y'].map({0.0: 0, 1.0: 1}, na_action='ignore')
            
            # For boolean features like 'Weekend', convert to 0/1
            for col in dataset_df.columns:
                if dataset_df[col].dtype == bool:
                    dataset_df[col] = dataset_df[col].map({False: 0, True: 1}, na_action='ignore')
        
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue
        
        # Save the processed dataset
        dataset_df.to_csv(os.path.join(dataset_dir, f"{dataset_name}_processed.csv"), index=False)
        
        # Prepare dataset results
        dataset_result = {
            'name': dataset_name,
            'size': len(dataset_df),
            'n_features': dataset_df.shape[1] - 1,  # Excluding target
            'meta_feature_stats': {}
        }
        
        # For each meta-feature
        for meta_feature in meta_features:
            print(f"Analyzing variability of {meta_feature}")
            
            # Calculate meta-feature variability
            try:
                variability, plot_file = plot_meta_feature_variability(
                    dataset_dir,
                    dataset_df,
                    meta_feature,
                    n_samples=n_subsets
                )
                
                # Store results
                dataset_result['meta_feature_stats'][meta_feature] = {
                    'variability': variability,
                    'plot_file': plot_file
                }
            except Exception as e:
                print(f"Error analyzing variability of {meta_feature}: {e}")
                continue
        
        # Add dataset results
        results['datasets'].append(dataset_result)
    
    # Save results
    try:
        with open(os.path.join(output_dir, 'variability_results.json'), 'w') as f:
            # Convert numpy values and arrays to standard types for JSON serialization
            def json_serialize(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (set, frozenset)):
                    return list(obj)
                elif isinstance(obj, complex):
                    return str(obj)
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                return str(obj)
            
            # Convert to JSON-serializable structure
            json_results = {}
            for key, value in results.items():
                if key == 'datasets':
                    json_results[key] = []
                    for dataset in value:
                        json_dataset = {k: v for k, v in dataset.items() if k != 'meta_feature_stats'}
                        json_dataset['meta_feature_stats'] = {}
                        
                        for mf, stats in dataset.get('meta_feature_stats', {}).items():
                            json_dataset['meta_feature_stats'][mf] = {}
                            for stat_key, stat_value in stats.items():
                                if stat_key == 'variability':
                                    json_dataset['meta_feature_stats'][mf][stat_key] = {}
                                    for var_key, var_value in stat_value.items():
                                        try:
                                            json_dataset['meta_feature_stats'][mf][stat_key][var_key] = json_serialize(var_value)
                                        except:
                                            json_dataset['meta_feature_stats'][mf][stat_key][var_key] = str(var_value)
                                else:
                                    json_dataset['meta_feature_stats'][mf][stat_key] = stat_value
                        
                        json_results[key].append(json_dataset)
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2, default=json_serialize)
    except Exception as e:
        print(f"Error saving results to JSON: {e}")
    
    return results

def create_summary_table(results, output_file=None):
    """
    Create a summary table of experiment results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_shift_convergence_experiment
    output_file : str
        File to save the table to (optional)
        
    Returns:
    --------
    pandas.DataFrame
        Summary table
    """
    # Extract data for table
    rows = []
    
    for exp in results['experiments']:
        
        row = {
            'Shift Type': exp['shift_type'],
            'Meta-feature': exp['meta_feature'],
            'Mutation Type': exp['mutation_type'],
            'Initial Value': exp['initial_value'],
            'Target Value': exp['target_value'],
            'Final Value': exp['final_value'],
            'Mean Initial Value': exp['mean_initial_value'],
            'Mean Target Value': exp['mean_target_value'],
            'Mean Final Value': exp['mean_final_value'],
            'Absolute Error': exp['absolute_error'],
            'Mean Absolute Error': exp['mean_absolute_error'],           
        }
        rows.append(row)
    
    # Create DataFrame
    summary_df = pd.DataFrame(rows)
    
    # Save to file if specified
    if output_file:
        summary_df.to_csv(output_file, index=False)
    
    return summary_df
def create_variability_summary(results, output_file=None):
    """
    Create a summary table of meta-feature variability.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_meta_feature_variability_experiment
    output_file : str
        File to save the table to (optional)
        
    Returns:
    --------
    pandas.DataFrame
        Summary table
    """
    # Extract data for table
    rows = []
    
    for dataset in results['datasets']:
        for meta_feature in results['meta_features']:
            if meta_feature in dataset['meta_feature_stats']:
                stats = dataset['meta_feature_stats'][meta_feature]['variability']
                
                if stats['mean_norm'] is not None:
                    row = {
                        'Dataset': dataset['name'],
                        'Meta-feature': meta_feature,
                        'Full Value': stats['full_value'],
                        'Mean': stats['mean_norm'],
                        'Std Dev': stats['std_norm'],
                        'CoV': stats['cov_norm'],
                        'Range': stats['range_norm'],
                    }
                    rows.append(row)
    
    # Create DataFrame
    summary_df = pd.DataFrame(rows)
    
    # Save to file if specified
    if output_file:
        summary_df.to_csv(output_file, index=False)
    
    return summary_df
def create_variability_summary1(results, output_file=None):
    """
    Create a summary table of meta-feature variability.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_meta_feature_variability_experiment
    output_file : str
        File to save the table to (optional)
        
    Returns:
    --------
    pandas.DataFrame
        Summary table
    """
    # Extract data for table
    rows = []
    
    for dataset in results['datasets']:
        for meta_feature in results['meta_features']:
            if meta_feature in dataset['meta_feature_stats']:
                stats = dataset['meta_feature_stats'][meta_feature]['variability']
                
                if stats['mean'] is not None:  # Check if we have valid stats
                    row = {
                        'Dataset': dataset['name'],
                        'Meta-feature': meta_feature,
                        'Full Value': stats['full_value'],
                        'Mean': stats['mean'],
                        'Std Dev': stats['std'],
                        'CoV': stats['cov'],  # Coefficient of Variation
                        'Range': stats['range'],
                    }
                    rows.append(row)
    
    # Create DataFrame
    summary_df = pd.DataFrame(rows)
    
    # Save to file if specified
    if output_file:
        summary_df.to_csv(output_file, index=False)
    
    return summary_df

if __name__ == "__main__":
    import os
    import glob
    from sklearn.preprocessing import LabelEncoder

    """
    Choose which data to use: 'real data' or 'synthetic data'
    Names of parametres to set:
        generations - number of generations
        n_trials - number of trials
        n_samples - number of samples
        scaling_factors - scaling factors
        mutation_prob - probability of mutation
    """

    experiment_type = 'real data'   #'real data' or 'synthetic data'


    if experiment_type == 'real data':
        # Define output directory for experiment results
        OUTPUT_DIR = 'real_data_experiments'
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        # Define meta-features to analyze
        META_FEATURES = ['eigenvalues', 'kurtosis', 'iq_range', 'cor', 'cov']
        
        # Define mutation types to test
        MUTATION_TYPES = ['row_noise', 'row_dist', 'row_cov', 'row_bn']
        
        print("\n===== DISTRIBUTION SHIFT ANALYSIS WITH REAL DATA =====\n")

        all_results = {}
        fl = 1
        if fl == 1:
            for i in ['MagicTelescope_fConc_']:
                dataset_name = i
                source_file = f'source/{dataset_name}.csv'
                target_file = f'target/{dataset_name}.csv'
                
                # Check if corresponding target file exists
                if not os.path.exists(target_file):
                    print(f"\nSkipping {dataset_name}: No matching target file found at {target_file}")
                    continue
                
                print(f"\n===== Running experiments for {dataset_name} dataset =====\n")
                
                # Create a directory for this dataset
                dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)
                
                # Run the shift convergence experiment
                try:
                    results = run_shift_convergence_experiment(
                        shift_type=f'file:{dataset_name}',  # Use file: prefix to indicate real data
                        meta_features=META_FEATURES,
                        mutation_types=MUTATION_TYPES,
                        n_trials=1,
                        output_dir=dataset_dir,
                        n_samples=500,  
                        scaling_factors=[1],
                        generations=50,  
                        source_file=source_file,
                        target_file=target_file
                    )
                    
                    all_results[dataset_name] = results
                    
                    # Create summary table
                    summary_df = create_summary_table(
                        results, 
                        output_file=os.path.join(dataset_dir, "summary.csv")
                    )
                    
                    print(f"\nSummary for {dataset_name}:")
                    print(summary_df.head())
                    
                except Exception as e:
                    print(f"Error running experiment for {dataset_name}: {e}")
                    import traceback
                    traceback.print_exc()
        # 3. Run meta-feature variability experiments
        print("\n===== Running meta-feature variability experiments =====\n")

        # Prepare list of datasets
        datasets = []
        for dataset_name in ['MagicTelescope_fConc_']:
            source_file = f'source/{dataset_name}.csv'
            datasets.append({
                'name': dataset_name,
                'file_path': source_file
            })
        
        # Run the variability experiment
        try:
            variability_dir = os.path.join(OUTPUT_DIR, "variability")
            if not os.path.exists(variability_dir):
                os.makedirs(variability_dir)
                
            variability_results = run_meta_feature_variability_experiment(
                datasets=datasets,
                meta_features=META_FEATURES,
                output_dir=variability_dir,
                n_subsets=20
            )
            
            # Create variability summary
            variability_summary = create_variability_summary(
                variability_results,
                output_file=os.path.join(variability_dir, "variability_summary.csv")
            )
            
            print("\nVariability summary:")
            if isinstance(variability_summary, pd.DataFrame) and not variability_summary.empty:
                print(variability_summary.head())
            else:
                print("No valid variability summary created.")
                
        except Exception as e:
            print(f"Error running variability experiments: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n===== All experiments completed =====")
        print(f"Results saved to {OUTPUT_DIR}")


    else: #if experiment_type == 'synthetic data':
        # Base output directory
        OUTPUT_DIR = "distribution_shift_results"
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        # Define meta-features to test
        META_FEATURES = [
            
            'cor',             # Correlation
            'eigenvalues',     # Eigenvalues of covariance matrix
            'kurtosis',        # Kurtosis
            'iq_range',        # Interquartile range
            'cov',             # Covariance
        ]
        
        # Define mutation types to test
        MUTATION_TYPES = [
            'row_bn',         # Row-wise Bayesian Network 
            'row_noise',       # Row-wise Gaussian noise
            'row_dist',        # Row-wise distribution sampling
            'row_cov',         # Row-wise covariance-based sampling
            'col_noise',       # Column-wise Gaussian noise
            'col_dist',        # Column-wise distribution sampling
        ]
        
        # Define distribution shift types
        SHIFT_TYPES = [
            'all',             # All the shifts together
            'covariate',       # Covariate shift
            'concept',         # Concept shift
            'prior',           # Prior probability shift
            'variance',        # Variance shift
            'correlation',     # Correlation shift
        ]
        
        # 1. Run convergence experiments for each shift type
        all_results = {}
        
        for shift_type in SHIFT_TYPES:
            print(f"\n===== Running experiments for {shift_type} shift =====\n")
            
            results = run_shift_convergence_experiment(
                shift_type=shift_type,
                meta_features=META_FEATURES,
                mutation_types=MUTATION_TYPES,
                n_trials=1,
                output_dir=OUTPUT_DIR,
                scaling_factors= [1],
                generations=100
            )
            
            all_results[shift_type] = results
            
            # Create summary table
            summary_df = create_summary_table(
                results, 
                output_file=os.path.join(OUTPUT_DIR, f"shift_{shift_type}", "summary.csv")
            )
            
            print(f"\nSummary for {shift_type} shift:")
            print(summary_df.head())
        
        # 2. Run meta-feature variability experiments
        # Define dataset generators
        dataset_generators1 = [
            # Each tuple is (generator_function, params_dict, dataset_name)
            (
                generate_shifted_classification_data,
                {'shift_type': 'all', 'n_samples': 500},
                'all_shifts'
            )]
        dataset_generators = [
            # Each tuple is (generator_function, params_dict, dataset_name)
            (
                generate_shifted_classification_data,
                {'shift_type': 'all', 'n_samples': 500},
                'all_shifts'
            ),
            (
                generate_isolated_shift_data,
                {'shift_type': 'covariate', 'n_samples': 500},
                'covariate_shift'
            ),
            (
                generate_isolated_shift_data,
                {'shift_type': 'concept', 'n_samples': 500},
                'concept_shift'
            ),
            (
                generate_isolated_shift_data,
                {'shift_type': 'prior', 'n_samples': 500},
                'prior_shift'
            ),
            (
                generate_isolated_shift_data,
                {'shift_type': 'variance', 'n_samples': 500},
                'variance_shift'
            ),
            (
                generate_isolated_shift_data,
                {'shift_type': 'correlation', 'n_samples': 500},
                'correlation_shift'
            ),
        ]
        
        print("\n===== Running meta-feature variability experiments =====\n")
        
        variability_results = run_meta_feature_variability_experiment(
            dataset_generators=dataset_generators,
            meta_features=META_FEATURES,
            output_dir=os.path.join(OUTPUT_DIR, "variability"),
            n_subsets=20
        )
        
        # Create variability summary
        variability_summary = create_variability_summary(
            variability_results,
            output_file=os.path.join(OUTPUT_DIR, "variability", "variability_summary.csv")
        )
        
        print("\nVariability summary:")
        print(variability_summary.head())
        
        print("\n===== All experiments completed =====")
        print(f"Results saved to {OUTPUT_DIR}") 
            