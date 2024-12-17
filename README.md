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
    PCMM requirements for EM estimation
    ```bash
    conda install scipy numpy
    pip install tqdm
    ```

    PCMM requirements for PyTorch estimation:
    ```bash
    conda install scipy pytorch cpuonly -c pytorch
    pip install tqdm
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
- **`paper/`**: Contains scripts for experiments and analysis pertaining to our paper (see paper/README.md)

---

## Example usage

### (Complex) diametrical clustering
The following lines of code will fit diametrical clustering to unit-norm data. If the input data is complex-valued, the estimated centroids will also be complex-valued. 
```python
from PCMM.phase_coherence_kmeans import diametrical clustering
X = ... # Data as np.array of size (n, p), either real or complex. X must be unit norm for each sample
K = 3  # Number of clusters to be inferred
num_repl = 1 # Number of K-means replications to choose the best model from (default 1)
max_iter = 10000 # Maximum number of iterations (default 10000)
tol = 1e-10 # Stopping tolerance for changes in the objective function between consecutive iterations. 
C,X_part,obj = diametrical_clustering(X,K,max_iter=10000,num_repl=num_repl,init=None,tol=1e-10)
```

This produces:
- **`C`**: Estimated centroids of size `(p,K)`.
- **`X_part`**: Data partition of size `(K, n)`.
- **`obj`**: Log-likelihood curve.

### Complex ACG mixture model EM estimation
To fit a **Complex ACG Mixture Model** with `K=3`, `p=10`, `rank=5`:
1. **Define the Model**:
    ```python
    from PCMM.PCMMnumpy import ACG
    X = ... # Data as np.array of size (observations, dimensions)
    K = 3  # Number of clusters to be inferred
    p = X.shape[1]  # Data dimensionality
    rank = 5  # Model rank (should be rank<=p)
    complex = np.any(np.iscomplex(X))
    params = None # params can be included to start estimation from a specific parameter setting
    model = ACG(K=K, p=p, rank=rank, complex=complex, params=params) 
    ```

2. **Run the Estimation**:
    ```python
    from PCMM.mixture_EM_loop import mixture_EM_loop
    params, posterior, loglik = mixture_EM_loop(
        model=model,
        data=X, 
        tol=1e-10,  # Tolerance for stopping estimation
        max_iter=100000,  # Maximum iterations
        num_repl=1,  # Independent replications from which the best estimate is selected
        init='dc++',  # Initialization method, here diametrical clustering ++
    )
    ```

    This produces:
    - **`params`**: Estimated model parameters.
    - **`posterior`**: Posterior probability matrix of size `(K, n)`.
    - **`loglik`**: Log-likelihood curve.

---


### Data specifications
The input data should always be an array of size either `nxp` (Watson, ACG, Normal, diametrical, least-squares) or `nxpxq` (MACG, SingularWishart, grassmann, weighted_grassmann). Here `n` corresponds to the number of observations, `p` is the data dimensionality (e.g., number of brain regions), and `q` is the number of frames in the orthonormal matrix (in our paper, q=2). Furthermore, the following restrictions must be met:
- For Watson, ACG, and diametrical clustering, the input data must be sample-wise unit norm.
- For Grassmann clustering and MACG, input samples must be `pxq` orthonormal matrices, i.e., each column is unit norm and orthogonal to all other columns. For example, the two eigenvectors of a cosine difference matrix. 
- For weighted Grassmann clustering and Singular Wishart, input samples are assumed to be `pxq` matrices, where the norm of each column corresponds to the square root of the associated eigenvalue. Thus, the sum of the norm of columns should equal the dimensionality `p`.
- The least-squares K-means algorithm implemented does a sign flip of input vectors under the hood, such that the majority of the elements of input vectors are negative.
- Only Watson, ACG, and diametrical_clustering are implemented to also handle complex-valued input data. 

### Initialization strategies:
- K-means models (implemented in `PCMM.phase_coherence_kmeans.py`) are by default initialized using their `++`-equivalents.
- Probabilistic mixture models may be initialized using K-means models:
    - `dc` = Diametrical clustering (for real or complex-valued input data)
    - `gc` = Grassmann clustering
    - `wgc` = Weighted Grassmann clustering
    - `ls` = least-squares clustering, where data is sign-flipped prior to clustering.
- Probabilistic mixture models may also be initialized using `K` one-component models on a partitioned part of the data, the partition given by one of the K-means or K-means++ models. For example, `init=dc_seg` first runs diametrical clustering, then estimates `K` different one-component models given the data partition. This strategy can be effective for high-rank models as well as PyTorch-estimated models that converge faster from a good seed. 
- Probabilistic mixture estimates may also be started using previously estimated parameters `params` of the same model. For ACG, MACG, Normal, and SingularWishart, this may be a lower-rank counterpart, e.g., `params['M'].shape = [pxr1]` and `options['rank']=r2`
- Hidden Markov models may also be initialized from the corresponding mixture model estimate. If so, the transition matrix will be initialized from a computed posterior probability matrix. If an HMM is initialized from a K-means model, the transition matrix is estimated from the empirical transition matrix. 

## High-level implementation

The PCMM folder also includes `helper_functions`, which include high-level train and test scripts for easy implementation across models. 
``` python
from PCMM.helper_functions import train_model,test_model
params,posterior,loglik = train_model(X_train,K,options,params,suppress_output,samples_per_sequence)
test_loglik,test_posterior,test_loglik_per_sample = test_model(X_test,params,K,options,samples_per_sequence)
```
The inputs for the train function are:
- `data_train` (input data specifications below)
- `K` (number of components to be estimated)
- `options` containing:
    - `options['modelname']` one of either `['Watson','Complex_Watson','ACG','Complex_ACG','MACG','SingularWishart','Normal','Complex_Normal','least_squares','diametrical','complex_diametrical','grassmann','weighted_grassmann']`
    - `options['rank']` one of either `['fullrank',r]` where `r<=p` is an integer (only for models with a covariance matrix parameter where this must be specified)
    - `options['LR']`, where 0 specifies EM estimation and any other value specifies PyTorch estimation learning rate (only for mixtures, not K-means)
    - `options['HMM']` (defaults to False) of either `[True,False]`, specifies whether to add HMM to the mixture model (only applicable to PyTorch mixture estimation)
    - `options['init']` of either `['uniform','dc','dc++','dc_seg','dc++_seg','gc','gc++','gc_seg','gc++_seg','wgc','wgc++','wgc_seg','wgc++_seg','ls','ls_seg']` (see description above)
    - `options['tol']` (defaults to 1e-10) Tolerance at which to stop estimation
    - `options['max_iter']` (defaults to 1e6) Maximum number of estimation loop iterations
    - `options['max_repl_inner']` (defaults to 1) Number of independent clustering replications to choose the best estimate from.
- `params` (defaults to None) Parameter set from which to start model estimation (e.g., a lower-rank equivalent of the model, see above)
- `suppress_output` (defaults to False) Whether to print estimated log-likelihood for each iteration
- `samples_per_sequence` (defaults to 0) Only for HMM - the number of samples in each sequence. Can be either an integer or a list of integers. If zero, it corresponds to all data being the same sequence. 

The inputs for the test function are the same, except `params` is a required input and `K` is no longer required. 


## Citing the work

If you use **PCMM** in your work, please cite:

**Uncovering dynamic human brain phase coherence networks**  
Anders S. Olsen, Anders Brammer, Patrick M Fisher, Morten Mørup (2024).  
Available at: [https://doi.org/10.1101/2024.11.15.623830](https://doi.org/10.1101/2024.11.15.623830)

---
