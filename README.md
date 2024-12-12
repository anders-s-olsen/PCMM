# Phase Coherence Mixture Modeling (PCMM)

**PCMM** is a Python-based toolbox designed to facilitate multivariate mixture modeling for multivariate dynamic phase coherence in functional brain imaging and beyond. This repository provides implementations of clustering using several statistical models, including the **Complex Angular Central Gaussian (ACG)** distribution, as well as analyses for both synthetic and real datasets pertaining to our paper [Uncovering dynamic human brain phase coherence networks](https://www.biorxiv.org/content/10.1101/2024.11.15.623830v1). 

## Table of Contents
1. [Setup Instructions](#setup-instructions)
2. [Repository Structure](#repository-structure)
3. [Using the Models](#using-the-models)
4. [Model specifications](#model-specifications)
5. [Citing this Repository](#citing-the-work)

---

## Setup Instructions

1. **Create a New Environment** (recommended):
    ```bash
    conda create -n PCMM python
    conda activate PCMM
    ```
    
2. **Install necessary dependencies**
    PCMM requirements (pytorch scripts only developed for cpu):
    ```bash
    conda install scipy pytorch cpuonly -c pytorch
    pip install tqdm
    ```
    Requirements for running analysis notebooks
    ```bash
    conda install ipykernel h5py matplotlib seaborn nilearn pandas 
    pip install networkx
    ```
    
4. **Clone the Repository**:
    ```bash
    git clone https://github.com/anders-s-olsen/PCMM.git
    cd PCMM
    ```

5. **Install the PCMM subfolder as a package**:
    - For standard installation:
        ```bash
        pip install .
        ```
    - For editable installation (allows for edits in the 'PCMM' folder):
        ```bash
        pip install -e .
        ```

---

## Repository Structure

- **`PCMM/`**: Contains the source code for the PCMM library, including model implementations and utility scripts.
- **`synthetic analysis/`**: Includes scripts for generating and analyzing synthetic datasets (manuscript Figs 1-3).
- **`experiments_phaserando/`**: Contains scripts and resources for running experiments involving phase-randomized data (manuscript Fig 4).
- **`experiments_real/`**: Includes scripts for processing and analyzing human connectome project (HCP) fMRI data (manuscript Figs 5-8).
- **`data/`**: Data preprocessing (filtering, GSR, phase calculation)
- **`atlas_figure/`**: Script for visualizing the Schaefer-100 atlas (manuscript Figs 5-6,8).

Only the **`PCMM/`** folder contains files necessary for implementation in other projects. The other folders contain analysis scripts and notebooks specific to generating the results in our manuscript. 

---

## Using the Models

### Input Requirements
Most models require input signals formatted as a NumPy array of shape `(observations, dimensions)`.

### Example Workflow using EM estimation
To fit a **Complex ACG Mixture Model** with `K=3`, `p=10`, `rank=5`:
1. **Define the Model**:
    ```python
    from PCMM.PCMM_EM.ACGEM import ACG_EM
    X = ... # Data
    K = 3  # Number of clusters to be inferred
    p = X.shape[1]  # Signal dimensionality
    rank = 5  # Model rank (should be rank<=p)
    model = ACG_EM(K=K, p=p, rank=rank, complex=True, params=None) #params can be included to start estimation from a specific parameter setting
    ```

2. **Set Estimation Options**:
    ```python
    options = {
        'tol': 1e-10,  # Tolerance for stopping estimation
        'max_iter': 100000,  # Maximum iterations
        'num_repl_inner': 1,  # Independent replications from which the best estimate is selected
        'init': 'dc++',  # Initialization method, here diametrical clustering ++
    }
    ```

3. **Run the Estimation**:
    ```python
    from PCMM.PCMM_EM.mixture_EM_loop import mixture_EM_loop

    params, posterior, loglik = mixture_EM_loop(
        model=model,
        data=X, 
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

## Model specifications

The PCMM folder also includes `helper_functions`, which include train and test scripts:
``` python
from PCMM.helper_functions import train_model,test_model
params,posterior,loglik = train_model(X,K,options,params,suppress_output,samples_per_sequence)
```
The inputs for the train function are:
- `data_train` (input data specifications below)
- `K` (number of components to be estimated)
- `options` containing:
    - `options['modelname']` one of either `['Watson','Complex_Watson','ACG','Complex_ACG','MACG','SingularWishart','Normal','Complex_Normal','euclidean','diametrical','complex_diametrical','grassmann','weighted_grassmann']`
    - `options['rank']` one of either `['fullrank',r]` where `r<=p` is an integer
    - `options['LR']`, where 0 specifies EM estimation and any other value specifies PyTorch estimation learning rate
    - `options['HMM']` of either `[True,False]`, only applicable to PyTorch estimation and specifies whether to add HMM to the mixture model
    - `options['init']` of either `['uniform','dc','dc++','dc_seg','dc++_seg','gc','gc++','gc_seg','gc++_seg','wgc','wgc++','wgc_seg','wgc++_seg','euclidean','euclidean_seg']`
    - `options['tol']` (defaults to 1e-10) Tolerance at which to stop estimation
    - `options['max_iter']` (defaults to 1e6) Maximum number of estimation loop iterations
    - `options['max_repl_inner']` (defaults to 1) Number of independent clustering replications to choose the best estimate from.
- `params` (defaults to None) Parameter set from which to start model estimation (e.g., a lower-rank equivalent of the model)
- `suppress_output` (defaults to False) Whether to print estimated log-likelihood for each iteration
- `samples_per_sequence` (defaults to 0) Only for HMM - the number of samples in each sequence. Can be either an integer or a list of integers. If zero, it corresponds to all data being the same sequence. 

The test script requires the estimated parameters as well as input:
``` python
test_loglik,test_posterior,test_loglik_per_sample = test_model(data_test,params,K,options)
```

### Data specifications
The input data should always be an array of size either `nxp` (Watson, ACG, Normal, diametrical, euclidean) or `nxpxq` (MACG, SingularWishart, grassmann, weighted_grassmann). Here `n` corresponds to the number of observations, `p` is the data dimensionality (e.g., number of brain regions), and `q` is the number of frames in the orthonormal matrix (in our paper, q=2). Only Watson, ACG, and diametrical_clustering are implemented to also handle complex-valued input data. 

### Initialization strategies:
- K-means models (implemented in `PCMM.riemannian_clustering.py`) are by default initialized using their `++`-equivalents.
- Probabilistic mixture models may be initialized using K-means models:
    - `dc` = Diametrical clustering (for real or complex-valued input data)
    - `gc` = Grassmann clustering
    - `wgc` = Weighted Grassmann clustering
    - `euclidean` = least-squares clustering, where data is sign-flipped prior to clustering.
- Probabilistic mixture models may also be implemented using previously estimated parameters `params` of the same model. For ACG, MACG, Normal, and SingularWishart, this may be a lower-rank counterpart, e.g., `params['M'].shape = [pxr1]` and `options['rank']=r2`
- Hidden Markov models may also be initialized from the corresponding mixture model estimate. If so, the transition matrix will be initialized from a computed posterior probability matrix.


## Citing the work

If you use **PCMM** in your work, please cite:

**Uncovering dynamic human brain phase coherence networks**  
Anders S. Olsen, Anders Brammer, Patrick M Fisher, Morten Mørup (2024).  
Available at: [https://doi.org/10.1101/2024.11.15.623830](https://doi.org/10.1101/2024.11.15.623830)

---

Feel free to customize this README further to match the repository's evolving features!
