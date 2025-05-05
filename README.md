# Vector-Valued Meta-Feature Analysis

This project provides tools for analyzing and manipulating vector-valued meta-features in datasets. It includes capabilities for distribution shift analysis, meta-feature variability assessment, and synthetic data generation through evolutionary algorithms.

## Setup

1. Create a virtual environment (recommended):
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

- `vector/run_experiments.py`: Main script to run distribution shift experiments
- `vector/vector_plots.py`: Visualization utilities for meta-feature analysis
- `vector/preprocess_fitness_mf.py`: Functions for computing meta-features and fitness
- `vector/mutations_crossover.py`: Genetic algorithm operators for data mutation
- `vector/shifts.py`: Functions to generate synthetic data with distribution shifts
- `vector/ChooseBN.py`: Utilities for Bayesian Network selection

## Data Organization

- Source datasets should be placed in the `source/` directory
- Target datasets should be placed in the `target/` directory

## Running Experiments

To run distribution shift experiments:

```
python vector/run_experiments.py
```

This will:
1. Load datasets from source/ and target/ directories
2. Analyze meta-feature variability
3. Run evolutionary algorithms to transform data between domains
4. Generate visualizations and summary tables

## Experiment Types

The script supports two types of experiments:
- Real data experiments: Using datasets from source/ and target/ directories
- Synthetic data experiments: Generating synthetic data with various distribution shifts

You can configure the experiment type in the main section of `run_experiments.py`.

## Meta-Features

The system analyzes several vector-valued meta-features:
- eigenvalues: Eigenvalues of the covariance matrix
- kurtosis: Measure of the "tailedness" of distributions
- iq_range: Interquartile range
- cor: Correlation coefficients
- cov: Covariance matrix elements

## Mutation Types

Various mutation strategies are available:
- row_noise: Add Gaussian noise to individual rows
- row_dist: Generate new values from marginal distributions
- row_cov: Generate new values based on covariance matrix
- row_bn: Use Bayesian Networks for generating realistic rows
- col_noise: Add Gaussian noise to entire columns
- col_dist: Replace columns with values from marginal distributions 