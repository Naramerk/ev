

# Row-wise mutation operators
def custom_mutate_noise(individual, mutation_prob=0.3, noise_scale=0.1, 
                        categorical_idx=None, continuous_idx=None, cat_probs=None, n_features=None):
    """
    Row-wise mutation: Add Gaussian noise to individual rows.
    
    Parameters:
    -----------
    individual : list
        Flattened data to mutate
    mutation_prob : float
        Probability of mutating each row
    noise_scale : float
        Scale of the Gaussian noise
    categorical_idx : list
        Indices of categorical features
    continuous_idx : list
        Indices of continuous features
    cat_probs : list
        Probability distributions for categorical features
    n_features : int
        Number of features
        
    Returns:
    --------
    tuple
        Mutated individual
    """
    # Reshape individual to 2D
    individual_data = np.array(individual).reshape(-1, n_features)
    mutated = individual_data.copy()
    
    # Mutate each row with probability mutation_prob
    for i in range(len(individual_data)):
        if random.random() > mutation_prob:
            continue
            
        # Add noise to continuous features
        if continuous_idx:
            # Scale noise relative to the data values
            noise_scale_adjusted = noise_scale * np.abs(mutated[i][continuous_idx])
            # Avoid zero noise scale
            noise_scale_adjusted = np.maximum(noise_scale_adjusted, 0.01)
            noise = np.random.normal(0, noise_scale_adjusted)
            
            # Ensure noise has correct dimensions
            if not isinstance(noise, np.ndarray) or len(noise) != len(continuous_idx):
                noise = np.random.normal(0, 0.1, len(continuous_idx))
                
            mutated[i][continuous_idx] += noise
            
        # Handle categorical features
        if categorical_idx and cat_probs:
            for j, cat_idx in enumerate(categorical_idx):
                if j < len(cat_probs) and random.random() < mutation_prob:
                    mutated[i, cat_idx] = np.random.choice(len(cat_probs[j]), p=cat_probs[j])
    
    # Replace any NaN values
    mutated = np.nan_to_num(mutated)
    if categorical_idx:
                            for cat_idx in categorical_idx:
                                # Round to nearest integer
                                mutated[:, cat_idx] = np.round(mutated[:, cat_idx]).astype(int)
                                
                                # Optional: ensure values are valid categories (assuming binary 0/1 classification)
                                mutated[:, cat_idx] = np.clip(mutated[:, cat_idx], 0, 1)
    return creator.Individual(mutated.flatten().tolist()),

def custom_mutate_dist(individual, mutation_prob=0.3, gmm=None, 
                       categorical_idx=None, continuous_idx=None, cat_probs=None, n_features=None):
    """
    Row-wise mutation: Generate new values from marginal distributions.
    
    Parameters:
    -----------
    individual : list
        Flattened data to mutate
    mutation_prob : float
        Probability of mutating each row
    gmm : GaussianMixture
        Gaussian Mixture model for sampling continuous features
    categorical_idx : list
        Indices of categorical features
    continuous_idx : list
        Indices of continuous features
    cat_probs : list
        Probability distributions for categorical features
    n_features : int
        Number of features
        
    Returns:
    --------
    tuple
        Mutated individual
    """
    # Reshape individual to 2D
    individual_data = np.array(individual).reshape(-1, n_features)
    mutated = individual_data.copy()
    
    # Mutate each row with probability mutation_prob
    for i in range(len(individual_data)):
        if random.random() > mutation_prob:
            continue
        
        # Generate new categorical values
        new_categorical = []
        if categorical_idx and cat_probs:
            new_categorical = [np.random.choice(len(p), p=p) for p in cat_probs]
            
        # Generate new continuous values
        new_continuous = []
        if continuous_idx and gmm:
            try:
                new_continuous = gmm.sample(1)[0].flatten()
                if len(new_continuous) > len(continuous_idx):
                    new_continuous = new_continuous[:len(continuous_idx)]
                elif len(new_continuous) < len(continuous_idx):
                    # Pad with zeros or repeat values if needed
                    new_continuous = np.pad(new_continuous, 
                                         (0, len(continuous_idx) - len(new_continuous)), 
                                         mode='constant')
            except Exception as e:
                print(f"Error in GMM sampling: {e}")
                # Fallback to using random normal distribution
                mean = np.mean(individual_data[:, continuous_idx], axis=0)
                std = np.std(individual_data[:, continuous_idx], axis=0)
                new_continuous = np.random.normal(mean, std)
        
        # Assemble new row
        if categorical_idx and continuous_idx:
            new_row = np.zeros(n_features)
            for j, idx in enumerate(categorical_idx):
                if j < len(new_categorical):
                    new_row[idx] = new_categorical[j]
            for j, idx in enumerate(continuous_idx):
                if j < len(new_continuous):
                    new_row[idx] = new_continuous[j]
        elif categorical_idx:
            new_row = np.zeros(n_features)
            for j, idx in enumerate(categorical_idx):
                if j < len(new_categorical):
                    new_row[idx] = new_categorical[j]
        else:
            new_row = np.zeros(n_features)
            for j, idx in enumerate(continuous_idx):
                if j < len(new_continuous):
                    new_row[idx] = new_continuous[j]
                    
        mutated[i] = new_row
    
    # Replace any NaN values
    mutated = np.nan_to_num(mutated)
    if categorical_idx:
                            for cat_idx in categorical_idx:
                                # Round to nearest integer
                                mutated[:, cat_idx] = np.round(mutated[:, cat_idx]).astype(int)
                                
                                # Optional: ensure values are valid categories (assuming binary 0/1 classification)
                                mutated[:, cat_idx] = np.clip(mutated[:, cat_idx], 0, 1)
    return creator.Individual(mutated.flatten().tolist()),
def custom_mutate_cov1(individual, mutation_prob=0.3,
                     categorical_idx=None, continuous_idx=None, cat_probs=None, n_features=None):
    """
    CMA-ES mutation
    
    Parameters:
    -----------
    individual : list
        Flattened data to mutate
    mutation_prob : float
        Probability of mutating each row
    categorical_idx : list
        Indices of categorical features
    continuous_idx : list
        Indices of continuous features
    cat_probs : list
        Probability distributions for categorical features
    n_features : int
        Number of features
        
    Returns:
    --------
    tuple
        Mutated individual
    """
    import cma
    import numpy as np
    import random
    
    # Reshape individual to 2D
    individual_data = np.array(individual).reshape(-1, n_features)
    mutated = individual_data.copy()
    
    try:
        # Calculate covariance matrix for continuous features
        if continuous_idx:
            continuous_data = individual_data[:, continuous_idx]
            # Ensure data is numerical and non-empty
            if continuous_data.size > 0 and not np.isnan(continuous_data).any():
                # Compute covariance matrix
                current_cov = np.cov(continuous_data.T)
                
                # Ensure it's a 2D matrix (for single feature case)
                if current_cov.ndim == 0:
                    current_cov = np.array([[current_cov]])
                
                # Ensure positive definiteness
                try:
                    min_eig = np.min(np.linalg.eigvals(current_cov))
                    if min_eig < 0:
                        current_cov -= 10*min_eig * np.eye(*current_cov.shape)
                except np.linalg.LinAlgError:
                    # If eigenvalue calculation fails, add a small positive value to diagonal
                    current_cov += 0.01 * np.eye(*current_cov.shape)
                
                # Generate new samples using CMA-ES
                try:
                    mean_vector = np.mean(continuous_data, axis=0)
                    # Initialize CMA-ES
                    sigma = 0.3  # Initial step size
                    opts = cma.CMAOptions()
                    opts.set('maxiter', 20)  # Limit iterations
                    opts.set('verbose', -9)  # Turn off output
                    
                    # Create evolution strategy from mean vector
                    es = cma.CMAEvolutionStrategy(mean_vector.tolist(), sigma, opts)
                    
                    # Generate population (same size as original data)
                    population = []
                    for _ in range(len(individual_data)):
                        es.tell(es.ask(), [0]*es.popsize)  # Dummy evaluation
                        population.append(es.result.xfavorite)
                    
                    new_continuous = np.array(population)
                    
                    # Apply with mutation probability
                    for i in range(len(individual_data)):
                        if random.random() <= mutation_prob:
                            # Update continuous features
                            mutated[i, continuous_idx] = new_continuous[i]
                except Exception as e:
                    print(f"Error generating samples with CMA-ES: {e}")
                    # Fallback to simple noise
                    for i in range(len(individual_data)):
                        if random.random() <= mutation_prob:
                            mutated[i, continuous_idx] += np.random.normal(
                                0, 0.1, len(continuous_idx)
                            )
        
        # Handle categorical features
        if categorical_idx and cat_probs:
            for i in range(len(individual_data)):
                if random.random() <= mutation_prob:
                    for j, cat_idx in enumerate(categorical_idx):
                        if j < len(cat_probs):
                            mutated[i, cat_idx] = np.random.choice(len(cat_probs[j]), p=cat_probs[j])
        
    except Exception as e:
        print(f"Error in CMA-ES mutation: {e}")
        # In case of error, add slight noise
        for i in range(len(individual_data)):
            if random.random() <= mutation_prob:
                mutated[i] += np.random.normal(0, 0.01, n_features)
    
    # Replace any NaN values
    mutated = np.nan_to_num(mutated)
    
    # Ensure categorical features are integers
    if categorical_idx:
        for cat_idx in categorical_idx:
            # Round to nearest integer
            mutated[:, cat_idx] = np.round(mutated[:, cat_idx]).astype(int)
            
            # Optional: ensure values are valid categories (assuming binary 0/1 classification)
            mutated[:, cat_idx] = np.clip(mutated[:, cat_idx], 0, 1)
            
    return creator.Individual(mutated.flatten().tolist()),
    

def custom_mutate_cov(individual, mutation_prob=0.3,
                     categorical_idx=None, continuous_idx=None, cat_probs=None, n_features=None):
    """
    Row-wise mutation: Generate new values based on covariance matrix.
    
    Parameters:
    -----------
    individual : list
        Flattened data to mutate
    mutation_prob : float
        Probability of mutating each row
    categorical_idx : list
        Indices of categorical features
    continuous_idx : list
        Indices of continuous features
    cat_probs : list
        Probability distributions for categorical features
    n_features : int
        Number of features
        
    Returns:
    --------
    tuple
        Mutated individual
    """
    # Reshape individual to 2D
    individual_data = np.array(individual).reshape(-1, n_features)
    mutated = individual_data.copy()
    
    try:
        # Calculate covariance matrix for continuous features
        if continuous_idx:
            continuous_data = individual_data[:, continuous_idx]
            # Ensure data is numerical and non-empty
            if continuous_data.size > 0 and not np.isnan(continuous_data).any():
                # Compute covariance matrix
                current_cov = np.cov(continuous_data.T)
                
                # Ensure it's a 2D matrix (for single feature case)
                if current_cov.ndim == 0:
                    current_cov = np.array([[current_cov]])
                
                # Ensure positive definiteness
                try:
                    min_eig = np.min(np.linalg.eigvals(current_cov))
                    if min_eig < 0:
                        current_cov -= 10*min_eig * np.eye(*current_cov.shape)
                except np.linalg.LinAlgError:
                    # If eigenvalue calculation fails, add a small positive value to diagonal
                    current_cov += 0.01 * np.eye(*current_cov.shape)
                
                # Generate new samples for continuous features using multivariate normal
                try:
                    mean_vector = np.mean(continuous_data, axis=0)
                    # Generate new data
                    new_continuous = np.random.multivariate_normal(
                        mean_vector, current_cov, size=len(individual_data)
                    )
                    
                    # Apply with mutation probability
                    for i in range(len(individual_data)):
                        if random.random() <= mutation_prob:
                            # Update continuous features
                            mutated[i, continuous_idx] = new_continuous[i]
                except Exception as e:
                    print(f"Error generating samples from covariance matrix: {e}")
                    # Fallback to simple noise
                    for i in range(len(individual_data)):
                        if random.random() <= mutation_prob:
                            mutated[i, continuous_idx] += np.random.normal(
                                0, 0.1, len(continuous_idx)
                            )
        
        # Handle categorical features
        if categorical_idx and cat_probs:
            for i in range(len(individual_data)):
                if random.random() <= mutation_prob:
                    for j, cat_idx in enumerate(categorical_idx):
                        if j < len(cat_probs):
                            mutated[i, cat_idx] = np.random.choice(len(cat_probs[j]), p=cat_probs[j])
        
    except Exception as e:
        print(f"Error in covariance mutation: {e}")
        # In case of error, add slight noise
        for i in range(len(individual_data)):
            if random.random() <= mutation_prob:
                mutated[i] += np.random.normal(0, 0.01, n_features)
    
    # Replace any NaN values
    mutated = np.nan_to_num(mutated)
    if categorical_idx:
                            for cat_idx in categorical_idx:
                                # Round to nearest integer
                                mutated[:, cat_idx] = np.round(mutated[:, cat_idx]).astype(int)
                                
                                # Optional: ensure values are valid categories (assuming binary 0/1 classification)
                                mutated[:, cat_idx] = np.clip(mutated[:, cat_idx], 0, 1)
    return creator.Individual(mutated.flatten().tolist()),

def custom_mutate_bn(individual, mutation_prob=0.3,
                      categorical_idx=None, continuous_idx=None, cat_probs=None, n_features=None,
                      bns=None):
        """Мутация с использованием байесовской сети"""
        # Skip if no valid Bayesian Networks
        if not bns:
            return custom_mutate_noise(individual)
        column_names = [f'x{i+1}' for i in range(n_features-1)] + ['y']
        # Преобразование в numpy array для работы с данными
        individual_data = np.array(individual).reshape(-1, 6)

        # Создаем DataFrame, если возможно с исходными названиями столбцов
        individual_df = pd.DataFrame(individual_data, columns=column_names)
            

        # Случайный выбор байесовской сети из пула
        if bns and len(bns) > 0:
            mutated_bn = np.random.choice(bns)

            # Число образцов для генерации новых данных
            num_samples = len(individual_df) // 4 + 1

            try:
                # Генерируем данные
                mutated_data = mutated_bn.sample(num_samples)
                #print('mutated_data.head(5)', mutated_data.head(5))

                # Ensure the generated data has the right columns
                # Create a new DataFrame with the required columns in the correct order
                # First ensure all required columns exist
                for col in column_names:
                    if col not in mutated_data.columns:
                        mutated_data[col] = 0  # Default value for missing columns
                
                # Then reindex to get columns in the correct order
                mutated_data = mutated_data[column_names]
                #print('mutated_data.head(5)1', mutated_data.head(5))
                print('individual_data', individual_data)

                # Check if we generated valid data
                if hasattr(mutated_data, 'empty') and not mutated_data.empty:
                    # Преобразуем к numpy array
                    if hasattr(mutated_data, 'to_numpy'):
                        mutated_np = mutated_data.to_numpy()
                    else:
                        mutated_np = np.array(mutated_data)

                    # Check if shape matches
                    if mutated_np.shape[1] == n_features:
                        mutated = individual_data.copy()

                        # Выбираем случайные строки для замены
                        indices_to_replace = random.sample(
                            range(len(mutated)),
                            min(len(mutated) // 4, len(mutated_np))
                        )

                        for i, idx in enumerate(indices_to_replace):
                            row_idx = i % len(mutated_np)
                            # Заменяем данные
                            mutated[idx] = mutated_np[row_idx]

                        # Replace NaN values
                        mutated = np.nan_to_num(mutated)
                        print('mutated.head(5)2', mutated)

                        # Ensure categorical values remain integers
                        if categorical_idx:
                            for cat_idx in categorical_idx:
                                # Round to nearest integer
                                mutated[:, cat_idx] = np.round(mutated[:, cat_idx]).astype(int)
                                
                                # Optional: ensure values are valid categories (assuming binary 0/1 classification)
                                mutated[:, cat_idx] = np.clip(mutated[:, cat_idx], 0, 1)
                        print('mutated.head(5)3', mutated)
                        return creator.Individual(mutated.flatten().tolist()),

            except Exception as e:
                print(f"Warning: BN mutation error: {str(e)[:100]}...")

        # If any errors or no BNs available, return with slight noise
        # noise = np.random.normal(0, 0.01, individual_data.shape)
        # mutated = individual_data + noise
        # mutated = np.nan_to_num(mutated)
        return creator.Individual(mutated.flatten().tolist()),
# Column-wise mutation operators
def column_mutate_noise(individual, mutation_prob=0.3, noise_scale=0.1,
                       categorical_idx=None, continuous_idx=None, cat_probs=None, n_features=None):
    """
    Column-wise mutation: Add Gaussian noise to entire columns.
    
    Parameters:
    -----------
    individual : list
        Flattened data to mutate
    mutation_prob : float
        Probability of mutating each column
    noise_scale : float
        Scale of the Gaussian noise
    categorical_idx : list
        Indices of categorical features
    continuous_idx : list
        Indices of continuous features
    cat_probs : list
        Probability distributions for categorical features
    n_features : int
        Number of features
        
    Returns:
    --------
    tuple
        Mutated individual
    """
    # Reshape individual to 2D
    individual_data = np.array(individual).reshape(-1, n_features)
    mutated = individual_data.copy()
    n_samples = len(individual_data)
    
    # Add noise to continuous columns with probability mutation_prob
    if continuous_idx:
        for j in continuous_idx:
            if random.random() <= mutation_prob:
                # Get column statistics
                col_mean = np.mean(mutated[:, j])
                col_std = np.std(mutated[:, j])
                
                # Generate noise for entire column
                column_noise = np.random.normal(0, noise_scale * col_std, n_samples)
                mutated[:, j] += column_noise
    
    # Handle categorical columns
    if categorical_idx and cat_probs:
        for idx, cat_idx in enumerate(categorical_idx):
            if idx < len(cat_probs) and random.random() <= mutation_prob:
                # Replace entire column with new samples
                mutated[:, cat_idx] = np.random.choice(
                    len(cat_probs[idx]), 
                    size=n_samples, 
                    p=cat_probs[idx]
                )
    
    # Replace any NaN values
    mutated = np.nan_to_num(mutated)
    
    return creator.Individual(mutated.flatten().tolist()),

def column_mutate_dist(individual, mutation_prob=0.3,
                      categorical_idx=None, continuous_idx=None, cat_probs=None, n_features=None):
    """
    Column-wise mutation: Replace entire columns with values from marginal distributions.
    
    Parameters:
    -----------
    individual : list
        Flattened data to mutate
    mutation_prob : float
        Probability of mutating each column
    categorical_idx : list
        Indices of categorical features
    continuous_idx : list
        Indices of continuous features
    cat_probs : list
        Probability distributions for categorical features
    n_features : int
        Number of features
        
    Returns:
    --------
    tuple
        Mutated individual
    """
    # Reshape individual to 2D
    individual_data = np.array(individual).reshape(-1, n_features)
    mutated = individual_data.copy()
    n_samples = len(individual_data)
    
    # Replace continuous columns with probability mutation_prob
    if continuous_idx:
        for j in continuous_idx:
            if random.random() <= mutation_prob:
                # Compute column statistics
                col_mean = np.mean(mutated[:, j])
                col_std = np.std(mutated[:, j])
                
                # Generate new column from normal distribution
                mutated[:, j] = np.random.normal(col_mean, col_std, n_samples)
    
    # Handle categorical columns
    if categorical_idx and cat_probs:
        for idx, cat_idx in enumerate(categorical_idx):
            if idx < len(cat_probs) and random.random() <= mutation_prob:
                # Replace entire column with new samples
                mutated[:, cat_idx] = np.random.choice(
                    len(cat_probs[idx]), 
                    size=n_samples, 
                    p=cat_probs[idx]
                )
    
    # Replace any NaN values
    mutated = np.nan_to_num(mutated)
    
    return creator.Individual(mutated.flatten().tolist()),

def column_mutate_shuffle(individual, mutation_prob=0.3,
                         categorical_idx=None, continuous_idx=None, cat_probs=None, n_features=None):
    """
    Column-wise mutation: Shuffle values within columns.
    
    Parameters:
    -----------
    individual : list
        Flattened data to mutate
    mutation_prob : float
        Probability of mutating each column
    categorical_idx : list
        Indices of categorical features
    continuous_idx : list
        Indices of continuous features
    cat_probs : list
        Probability distributions for categorical features
    n_features : int
        Number of features
        
    Returns:
    --------
    tuple
        Mutated individual
    """
    # Reshape individual to 2D
    individual_data = np.array(individual).reshape(-1, n_features)
    mutated = individual_data.copy()
    
    # Get all column indices
    all_cols = list(range(n_features))
    
    # Shuffle selected columns
    for j in all_cols:
        if random.random() <= mutation_prob:
            # Get column and shuffle it
            col = mutated[:, j].copy()
            np.random.shuffle(col)
            mutated[:, j] = col
    
    # Replace any NaN values
    mutated = np.nan_to_num(mutated)
    
    return creator.Individual(mutated.flatten().tolist()),

def custom_crossover(ind1, ind2, cxpb=0.6, row_mode_prob=0.5, n_features=None):
    """
    Custom crossover that can work on rows or columns.
    
    Parameters:
    -----------
    ind1, ind2 : list
        Individuals to cross
    cxpb : float
        Crossover probability
    row_mode_prob : float
        Probability of performing row-wise crossover (vs column-wise)
    n_features : int
        Number of features
        
    Returns:
    --------
    tuple
        Crossed individuals
    """
    if random.random() >= cxpb:
        return ind1, ind2
    
    # Reshape to 2D
    matrix1 = np.array(ind1).reshape(-1, n_features)
    matrix2 = np.array(ind2).reshape(-1, n_features)
    n_samples = matrix1.shape[0]
    
    # Choose crossover type (row or column)
    if random.random() < row_mode_prob:
        # Row-wise crossover
        # Select random rows to exchange
        n_rows = int(n_samples * 0.3)  # Exchange about 30% of rows
        rows = random.sample(range(n_samples), k=n_rows)
        
        # Exchange rows
        temp = matrix1[rows].copy()
        matrix1[rows] = matrix2[rows]
        matrix2[rows] = temp
    else:
        # Column-wise crossover
        # Select random columns to exchange
        n_cols = int(n_features * 0.3)  # Exchange about 30% of columns
        cols = random.sample(range(n_features), k=n_cols)
        
        # Exchange columns
        temp = matrix1[:, cols].copy()
        matrix1[:, cols] = matrix2[:, cols]
        matrix2[:, cols] = temp
    
    # Replace any NaN values
    matrix1 = np.nan_to_num(matrix1)
    matrix2 = np.nan_to_num(matrix2)
    
    # Convert back to flattened list
    ind1[:] = matrix1.flatten().tolist()
    ind2[:] = matrix2.flatten().tolist()
    
    return ind1, ind2