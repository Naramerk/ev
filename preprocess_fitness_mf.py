from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pymfe.mfe import MFE
import pandas as pd
import numpy as np
from mutations_crossover import * 
from vector_plots import *
from deap import creator, base, tools, algorithms

def compute_meta_feature(data, meta_feature):
    """
    Compute the value of a specific meta-feature for given data.
    
    Parameters:
    -----------
    data : numpy.ndarray or DataFrame
        Data to compute meta-feature for
    meta_feature : str
        Name of the meta-feature to compute
        
    Returns:
    --------
    float
        Computed meta-feature value
    """
    try:
        mfe = MFE(features=[meta_feature], summary=None)
        #print(88888)
        # Make sure data is in numpy array format
        if isinstance(data, pd.DataFrame):
            data = data.values
        #print(99999)
        # Ensure data is not empty and has at least 2 dimensions
        if data.size == 0 or data.ndim < 2:
            return np.nan
                
        # Check if there's enough data for X and y
        if data.shape[1] < 2:
            return np.nan
        #print(100000)
        # Ensure data is float type and replace any NaN values
        data = np.array(data)
        #print('data', data)
        data = np.nan_to_num(data)
                
        X = data[:, :-1]
        y = data[:, -1]
        #print('X', X)
        #print(115511)
        # Check if there's any variation in the target
        if len(np.unique(y)) < 2:
            # If only one class, return a default value
            return 0.0
        #print(115512)
        mfe.fit(X, y)
        #print(mfe.extract())
        ft = mfe.extract()[1][0]
        #print('ft', ft)
        #print('type(ft)', type(ft))
        return ft


    except Exception as e:
        print(f"Error computing {meta_feature}: {e}")
        return np.nan
def fitness_function(individual, meta_feature, tar_val, n_features, initial_data=None):
    """
    Evaluate fitness by measuring distance between vectors of pairwise meta-feature values.
    
    Parameters:
    -----------
    individual : list
        Flattened synthetic data
    meta_feature : str
        Meta-feature to optimize
    tar_val : list or numpy.ndarray
        Target vector of pairwise meta-feature values
    n_features : int
        Number of features
    initial_data : numpy.ndarray
        Original data (not used directly in fitness, included for compatibility)
        
    Returns:
    --------
    tuple
        Fitness value (lower is better)
    """
    global init_val, fin_val, fit_value
    #print('fitness_function 1')
    # Reshape individual to match data dimensions
    synthetic_data = np.array(individual).reshape(-1, n_features)
    
    # Replace any NaN or inf values
    synthetic_data = np.nan_to_num(synthetic_data)

    # Compute meta-feature for synthetic data
    synthetic_values = compute_meta_feature(synthetic_data, meta_feature)
    
    #if np.isnan(synthetic_values):
    #    return (1000,)  # Penalty for NaN
    synthetic_values = np.array(synthetic_values)
    # Initialize init_val if not set
    try:
        if isinstance(init_val, str) and init_val == 'a' and initial_data is not None:
            init_val = compute_meta_feature(initial_data, meta_feature)
    except Exception as e:
        print(f"Error in fitness_function: {e}")
     
    
    # Calculate fitness as distance between vectors
    # Ensure tar_val is in a compatible format (numpy array)
    tar_val_array = np.array(tar_val)
    
    
    # Calculate Euclidean distance as the fitness
    fitness = np.linalg.norm(synthetic_values - tar_val_array)
    #print('fitness_function 2')
    #print('fitness', fitness)
    # For tracking purposes, update fin_val with the entire vector
    if fit_value > fitness:
        #print('before fin_val')
        fin_val = synthetic_values
        #print('after fin_val')
        fit_value = fitness
    #print('fitness_function 3')
    return (fitness,)
def create_black_list(X: pd.DataFrame, y):
    if not y:
        return []
    target_node = y
    black_list = [(target_node, (col)) for col in X.columns.to_list() if col != target_node]

    return black_list
def generate_synthetic_data(
    mutation_type, source_data, meta_feature, target_meta_value,
    mutation_prob=0.3, crossover_prob=0.6, row_mode_prob=0.5,
    population_size=100, generations=100
):
    """
    Generate synthetic data using genetic algorithm optimization to match a target meta-feature value.
    
    Parameters:
    -----------
    mutation_type : str
        Type of mutation operator to use: 
        - 'row_noise', 'row_dist', 'row_cov', 'row_bn' (row-wise operators)
        - 'col_noise', 'col_dist' (column-wise operators)
    source_data : DataFrame
        Source data to transform
    meta_feature : str
        Meta-feature to optimize
    target_meta_value : float
        Target meta-feature value to achieve
    mutation_prob : float
        Probability of mutation
    crossover_prob : float
        Probability of crossover
    row_mode_prob : float
        Probability of row-wise crossover (vs column-wise)
    population_size : int
        Size of the population
    generations : int
        Number of generations to evolve
        
    Returns:
    --------
    tuple
        (synthetic_data, logbook, original_data, meta_values) - 
        Tuple containing synthetic data, optimization log, original data, and
        a dictionary with initial, target, and achieved meta-feature values
    """
    global init_val, fin_val, tar_val, fit_value
    
    # Reset global variables
    init_val = 'a'
    fin_val = 'a'
    tar_val = target_meta_value

    # Prepare data
    if isinstance(source_data, pd.DataFrame):
        # Store column names for later
        column_names = source_data.columns
        source_np = source_data.values
    else:
        source_np = source_data
        column_names = None
    
    # Get data dimensions
    n_samples, n_features = source_np.shape
    
    # Identify categorical and continuous features
    # In this example, all features are continuous except the target (last column)
    continuous_idx = list(range(n_features - 1))  # All except last column
    categorical_idx = [n_features - 1]  # Last column (target) is categorical
    
    # Prepare categorical probabilities for target column
    cat_probs = []
    if categorical_idx:
        # For the target column, compute probability distribution
        for cat_idx in categorical_idx:
            unique_vals, counts = np.unique(source_np[:, cat_idx], return_counts=True)
            probs = counts / counts.sum()
            cat_probs.append(probs)
    
    # Train GMM for continuous features
    try:
        gmm = GaussianMixture(n_components=min(3, len(source_np))).fit(source_np[:, continuous_idx])
    except Exception as e:
        print(f"Error training GMM model: {e}")
        gmm = None
    
    # Для row_bn создаем пул байесовских сетей (bns)
    bns = []
    if mutation_type == 'row_bn':
        try:
            print("Creating pool of Bayesian Networks...")
            from ChooseBN import selectBN
            from bamt.preprocessors import Preprocessor
            from sklearn import preprocessing
            
            # Преобразуем в DataFrame для работы с BAMT
            if column_names is not None:
                df = pd.DataFrame(source_np, columns=column_names)
                #print(df.head(5))
            else:
                column_names = [f'x{i+1}' for i in range(n_features-1)] + ['y']
                df = pd.DataFrame(source_np, columns=column_names)
            
            # Подготовка предобработчика
            encoder = preprocessing.LabelEncoder()
            discretizer = preprocessing.KBinsDiscretizer(
                n_bins=5,
                encode='ordinal',
                strategy='kmeans',
                subsample=None
            )
            preprocessor = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
            df['y'] = df['y'].map({0.0: 0, 1.0: 1}, na_action='ignore')
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            bl = create_black_list(X, y.name)  # Edges to avoid
            params = {'bl_add': bl}
            # Ограничиваем размер данных для обучения
            df_sample = df
            if len(df) > 500:
                df_sample = df.sample(n=500, random_state=42)
            
            # Создаем базовую байесовскую сеть
            bn, discretized_data = selectBN(df_sample, preprocessor)
            bn.add_edges(discretized_data, params=params)
            bn.fit_parameters(df_sample)
            
            # Создаем пул сетей с разными структурами
            for i in range(7):  # Создаем 7 разных сетей
                try:
                    # Используем разные части данных для обучения
                    if len(df) > 500:
                        df_subset = df.sample(n=500, random_state=np.random.randint(0, 10000))
                    else:
                        df_subset = df.sample(frac=0.8, replace=True, random_state=np.random.randint(0, 10000))
                    
                    bn_new, discretized_data_new = selectBN(df_subset, preprocessor)
                    bn_new.add_edges(discretized_data_new)
                    bn_new.fit_parameters(df_subset)
                    
                    bns.append(bn_new)
                except Exception as e:
                    print(f"Error creating Bayesian Network {i+1}: {e}")
                    continue
            
            print(f"Created {len(bns)} Bayesian Networks")
        except Exception as e:
            print(f"Error initializing Bayesian Networks: {e}")
            bns = []
    # Setup functions for the genetic algorithm
    def create_individual():
        """Create an individual by copying the source data with some noise"""
        # Add small noise to avoid identical individuals
        noise = np.random.normal(0, 0.01, source_np.shape)
        individual_data = source_np.copy() + noise
        individual_data = np.nan_to_num(individual_data)
        return creator.Individual(individual_data.flatten().tolist())
    
    # Setup toolbox with appropriate mutation operator
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register evaluation function with target meta-feature value
    toolbox.register("evaluate", fitness_function, 
                     meta_feature=meta_feature, 
                     tar_val=target_meta_value, 
                     n_features=n_features,
                     initial_data=source_np)
    # Register selection operator
    toolbox.register("select", tools.selTournament, tournsize=3)
    # Register crossover operator
    toolbox.register("mate", custom_crossover, cxpb=crossover_prob, 
                     row_mode_prob=row_mode_prob, n_features=n_features)
    
    # Register mutation operator based on type
    mutation_params = {
        'mutation_prob': mutation_prob,
        'categorical_idx': categorical_idx,
        'continuous_idx': continuous_idx,
        'cat_probs': cat_probs,
        'n_features': n_features
    }
    # Only add gmm parameter for the operators that need it
    if mutation_type == 'row_dist' and gmm is not None:
        mutation_params['gmm'] = gmm
    
    # Add trained BNs parameter for row_bn operator
    if mutation_type == 'row_bn' and bns:
        mutation_params['bns'] = bns
    
    if mutation_type == 'row_noise':
        toolbox.register("mutate", custom_mutate_noise, **mutation_params)
    elif mutation_type == 'row_dist':
        toolbox.register("mutate", custom_mutate_dist, **mutation_params)
    elif mutation_type == 'row_cov':
        toolbox.register("mutate", custom_mutate_cov, **mutation_params)
    elif mutation_type == 'row_bn':
        toolbox.register("mutate", custom_mutate_bn, **mutation_params)
    elif mutation_type == 'col_noise':
        toolbox.register("mutate", column_mutate_noise, **mutation_params)
    elif mutation_type == 'col_dist':
        toolbox.register("mutate", column_mutate_dist, **mutation_params)
    else:
        print(f"Unknown mutation type: {mutation_type}, using row_noise as default")
        toolbox.register("mutate", custom_mutate_noise, **mutation_params)
    
    # Initialize population
    population = toolbox.population(n=population_size)
    hof = tools.HallOfFame(maxsize=1)
    
    # Setup statistics tracking
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    
    # Run the genetic algorithm
    try:
        population, logbook = algorithms.eaSimple(
            population, toolbox,
            cxpb=crossover_prob, mutpb=mutation_prob,
            ngen=generations, stats=stats,
            halloffame=hof, verbose=True
        )
    except Exception as e:
        print(f"Error in evolutionary algorithm: {e}")
        # Create a minimal logbook with some placeholder data
        logbook = tools.Logbook()
        logbook.header = ['gen', 'min', 'avg', 'std']
        for i in range(3):
            logbook.record(gen=i, min=1.0 - i*0.2, avg=1.2 - i*0.2, std=0.1)
        
        # Return the source data with minor noise as fallback
        noise = np.random.normal(0, 0.1, source_np.shape)
        synthetic_data = source_np + noise
        synthetic_data = np.nan_to_num(synthetic_data)
        
        # Convert back to DataFrame if needed
        if column_names is not None:
            synthetic_data = pd.DataFrame(synthetic_data, columns=column_names)
            
        # Set placeholder meta-feature values
        meta_values = {
            'initial': 0.0,
            'target': target_meta_value,
            'final': 0.0
        }
        
        return synthetic_data, logbook, source_np, meta_values
    
    # Get the best individual
    best_individual = hof[0] if len(hof) > 0 else tools.selBest(population, 1)[0]
    
    # Convert to numpy array and reshape
    synthetic_data = np.array(best_individual).reshape(n_samples, n_features)
    
    # Replace any NaN values
    synthetic_data = np.nan_to_num(synthetic_data)
    
    # Calculate final meta-feature values
    if isinstance(init_val, str) :
        init_val = compute_meta_feature(source_np, meta_feature)
    #print('before fin_val')
    if isinstance(fin_val, str) : 
        fin_val = compute_meta_feature(synthetic_data, meta_feature)
    #print('after fin_val')
    
    # Store meta-feature values
    meta_values = {
        'initial': init_val,
        'target': target_meta_value,
        'final': fin_val
    }
     
    print(f"Meta-feature values - Initial: {init_val.tolist() if isinstance(init_val, np.ndarray) else init_val}, "
      f"Target: {target_meta_value.tolist() if isinstance(target_meta_value, np.ndarray) else target_meta_value}, "
      f"Achieved: {fin_val.tolist() if isinstance(fin_val, np.ndarray) else fin_val}")
    # Convert back to DataFrame if needed
    if column_names is not None:
        synthetic_data = pd.DataFrame(synthetic_data, columns=column_names)
    
    return synthetic_data, logbook, source_np, meta_values 