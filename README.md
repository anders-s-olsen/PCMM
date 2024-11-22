# Phase Coherence Mixture Modeling (PCMM)

**PCMM** is a Python-based toolbox designed to facilitate multivariate mixture modeling for dynamic phase coherence analysis in functional brain imaging and beyond. This repository provides implementations of statistical models, including the **Complex Angular Central Gaussian (ACG)** distribution, tailored for analyzing both synthetic and real datasets. The framework offers tools for robust probabilistic clustering and Hidden Markov modeling (HMM) of complex-valued phase coherence data.

## Table of Contents
1. [Setup Instructions](#setup-instructions)
2. [Repository Structure](#repository-structure)
3. [Using the Models](#using-the-models)
4. [Data Requirements](#data-requirements)
5. [Citing this Repository](#citing-this-repository)

---

## Setup Instructions

1. **Create a New Environment** (recommended):
    ```bash
    conda create -n PCMM python=3.9
    conda activate PCMM
    ```

2. **Clone the Repository**:
    ```bash
    git clone https://github.com/anders-s-olsen/PCMM.git
    cd PCMM
    ```

3. **Install the Package**:
    - For standard installation:
        ```bash
        pip install .
        ```
    - For editable installation (allows for edits):
        ```bash
        pip install -e .
        ```

---

## Repository Structure

- **`src/`**: Contains the source code for the PCMM library, including model implementations and utility scripts.
- **`experiments_phaserando/`**: Contains scripts and resources for running synthetic experiments involving random phase data.
- **`experiments_real/`**: Includes scripts for processing real-world neuroimaging data.
- **`synthetic analysis/`**: Focused on generating and analyzing synthetic datasets.
- **`data/`**: A guide to obtaining, preprocessing, and using datasets. See `data/README.md` for more details.
- **`atlas_figure/`**: Scripts and outputs for creating atlas-based visualizations.

Each folder includes its own `README.md` file, describing its contents and usage in greater detail.

---

## Using the Models

### Input Requirements
The models require input signals formatted as a NumPy array of shape `(observations, dimensions)`.

### Example Workflow
To fit a **Complex ACG Mixture Model** with `K=3`, `p=10`, `rank=5`:
1. **Define the Model**:
    ```python
    from pcmm.models import ACG_EM

    K = 3  # Number of clusters
    p = 10  # Signal dimensionality
    rank = 5  # Model rank
    model = ACG_EM(K=K, p=p, rank=rank, complex=True)
    ```

2. **Set Estimation Options**:
    ```python
    options = {
        'tol': 1e-10,  # Tolerance for stopping estimation
        'max_iter': 100000,  # Maximum iterations
        'LR': 0,  # 0 for EM estimation; >0 for PyTorch-based
        'HMM': False,  # Enable Hidden Markov estimation (PyTorch only)
        'num_repl_inner': 1,  # Independent replications
        'init': '++',  # Initialization method
    }
    ```

3. **Run the Estimation**:
    ```python
    from pcmm.estimation import mixture_EM_loop

    params, posterior, loglik = mixture_EM_loop(
        model, data_train, 
        tol=options['tol'], 
        max_iter=options['max_iter'], 
        num_repl=options['num_repl_inner'], 
        init=options['init']
    )
    ```

    This produces:
    - **`params`**: Estimated model parameters.
    - **`posterior`**: Posterior probability matrix of size `(K, n)`.
    - **`loglik`**: Log-likelihood curve.

---

## Data Requirements

Refer to the `data/README.md` file for instructions on downloading and preprocessing datasets.

---

## Citing this Repository

If you use **PCMM** in your work, please cite:

**Uncovering dynamic human brain phase coherence networks**  
Anders S. Olsen et al. (2024).  
Available at: [https://doi.org/10.1101/2024.11.15.623830](https://doi.org/10.1101/2024.11.15.623830)

---

Feel free to customize this README further to match the repository's evolving features!
