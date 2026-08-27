# Phase Coherence Mixture Modeling (PCMM)

PCMM is a Python toolbox for clustering phases and representations derived from phases. The repository includes K-means-type algorithms, mixture models based on distributions from directional statistics, and hidden Markov models (HMMs). The repository provides a NumPy/SciPy route for K-means and expectation maximization (EM) fitting of mixture models, and a PyTorch route based on gradient optimization for fitting mixture models and HMMs.

This repo was introduced alongside our PNAS-paper [Uncovering dynamic human brain phase coherence networks](https://www.pnas.org/doi/10.1073/pnas.2518287123) but also contains work-in-progress extensions. 

## Phase representations and models

The PNAS-paper focuses on clustering phase representstions in multivariate functional neuroimaging data. We showed that modeling phases using the complex ACG mixture provides a clean representation free from amplitude-noise (such as motion), which improved in task recognition over Gaussian mixtures for the (filtered) time-series and complex Gaussian for the analytic signal. We also compared our models to leading-eigenvector-of-cosine-phase-coherence-matrix representations, also known as LEiDA. 

The progressively more detailed worked example is in [`tutorial.ipynb`](tutorial.ipynb). Code and dependencies used only to reproduce the PNAS-paper are documented in [`paper/`](paper/README.md).

## Contents

1. [Installation](#installation)
2. [Data specifications](#data-specifications)
3. [Models used in the paper](#models-used-in-the-paper)
4. [Direct model use](#direct-model-use)
5. [High-level functions](#high-level-functions)
6. [Initialization](#initialization)
7. [Hidden Markov models](#hidden-markov-models)
8. [Complete options reference](#complete-options-reference)
9. [Returned values and parameters](#returned-values-and-parameters)
10. [Additional models: work in progress](#additional-models-work-in-progress)
11. [Repository structure](#repository-structure)
12. [Citing the work](#citing-the-work)

## Installation

To create an environment, clone the repository, and install PCMM:

```bash
conda create -n PCMM python
conda activate PCMM
git clone https://github.com/anders-s-olsen/PCMM.git
cd PCMM
conda install numpy scipy
pip install tqdm
pip install .
```

For an editable installation, replace `pip install .` by `pip install -e .`. NumPy and SciPy are sufficient for non-PyTorch model fitting. Install PyTorch only if gradient-based optimization or HMMs are required (see the [PyTorch website](https://pytorch.org/) for latest install instructions):

```bash
conda install pytorch
```

Please note, GPU should be supported but has not been tested. Extra packages needed by the paper are listed only in [`paper/README.md`](paper/README.md).

## Data specifications

Let `n` be the number of observations and `p` the number of variables, channels, or brain regions. The paper methods use the following arrays:

| Representation | Shape | Meaning | Models |
|---|---:|---|---|
| Real vectors | `(n, p)` | Unnormalized observations, such as time series | real Normal |
| Complex vectors | `(n, p)` | Unnormalized analytic signals | complex Normal |
| Real projective vectors (LEiDA) | `(n, p)` | Unit norm and invariant to `x -> -x` | real diametrical, real ACG |
| Complex projective vectors | `(n, p)` | Unit norm and invariant to `x -> exp(1j * alpha) * x` | complex diametrical, complex ACG |

PCMM also contains methods for orthonormal frames, scaled frames, and phase angles on a torus. Those input formats and their implementations are described under [Additional models: work in progress](#additional-models-work-in-progress).

### Models used in the paper

| Method | Real data | Complex data | Fitting route |
|---|---|---|---|
| Diametrical clustering | real projective vectors | complex projective vectors | direct clustering |
| ACG mixture | real projective vectors | complex projective vectors | NumPy/EM or PyTorch |
| Normal mixture | real vectors | complex analytic signals | NumPy/EM or PyTorch |

All mixtures/HMMs have a low-rank implementation which parameterizes covariance-type parameters through an identity term and a low-rank factor `M`:

$$
\Psi_k = M_k M_k^{\mathrm H} + \gamma_k I,
$$

where `M` has shape `(K, p, rank)`. NumPy also provides a direct full-covariance `Psi` parameterization. Statistical distribution parameters in Numpy are fitted using a fixed-point algorithm while in PyTorch they are fitted using stochastic gradient optimization, thereby being faster.

## Direct model use

### Diametrical clustering

The same function handles real and complex data according to the input dtype:

```python
from PCMM.phase_coherence_kmeans import diametrical_clustering

C, labels, objective = diametrical_clustering(X, K=3, max_iter=10000, num_repl=5, init='++', tol=1e-10, suppress_output=False)
```

`C` has shape `(K, p)`, `labels` has shape `(n,)`, and `objective` records the retained replication's objective. `num_repl` indicates the number of independently seeded fits, where the output reflects the best fit. Similarity in diametrical clustering is based on `abs(X @ C.conj().T) ** 2`, so a sign flip or global complex phase does not change an assignment.

### Mixtures: NumPy/EM or PyTorch

The NumPy constructors are:

```python
from PCMM.PCMMnumpy import ACG, Normal, Watson

acg = ACG(p=p, rank=rank, K=K, complex=False, params=None)
complex_acg = ACG(p=p, rank=rank, K=K, complex=True, params=None)
normal = Normal(p=p, rank=rank, K=K, complex=False, params=None)
complex_normal = Normal(p=p, rank=rank, K=K, complex=True, params=None)
```

For NumPy ACG and Normal models, `rank=None` or `rank=0` uses a full `Psi` matrix. A positive integer `rank` uses the factorized parameterization. Use `params` to supply initial parameters (or use `init` in the fitting function below).

Fit a NumPy model using EM:

```python
from PCMM.mixture_EM_loop import mixture_EM_loop

params, posterior, loglik = mixture_EM_loop(model, X, tol=1e-8, max_iter=10000, num_repl=1, init='dc', suppress_output=False, num_comparison=10)
```

Here, `init` indicates the ([initializer to use](#initialization)), `suppress_output` suppresses `tqdm`, and `num_comparison` indicates the number of iterations compared in the early stopping criterion. 

The corresponding PyTorch constructors use a positive integer rank and can optionally enable an HMM:

```python
from PCMM.PCMMtorch import ACG, Normal, Watson

acg = ACG(p=p, rank=rank, K=K, HMM=False, complex=False, samples_per_sequence=0, params=None)
complex_acg = ACG(p=p, rank=rank, K=K, HMM=False, complex=True, samples_per_sequence=0, params=None)
normal = Normal(p=p, rank=rank, K=K, HMM=False, complex=False, samples_per_sequence=0, params=None)
complex_normal = Normal(p=p, rank=rank, K=K, HMM=False, complex=True, samples_per_sequence=0, params=None)
```

`samples_per_sequence` matters only for HMMs and is explained under [Hidden Markov models](#hidden-markov-models). Fit a PyTorch model with Adam by supplying a learning rate:

```python
from PCMM.mixture_torch_loop import mixture_torch_loop

params, posterior, loglik = mixture_torch_loop(model, X, tol=1e-8, max_iter=100000, num_repl=1, init='dc', LR=0.1, suppress_output=False, threads=8, decrease_lr_on_plateau=False, num_comparison=50)
```

## High-level functions

`train_model` and `test_model` provide one interface across clustering, EM, and PyTorch models. 

```python
from PCMM.helper_functions import train_model, test_model

options = {
    'modelname': 'Complex_ACG',
    'rank': 5,
    'LR': 0,
    'init': 'dc',
    'tol': 1e-10,
    'max_iter': 100000,
    'num_repl': 1,
    'HMM': False,
}

params, posterior, loglik = train_model(data_train=X_train, K=3, params=None, options=options)
test_loglik, test_posterior, test_loglik_per_sample = test_model(data_test=X_test, params=params, K=3, options=options)
```

Set `LR=0` for the NumPy/EM implementation or set `LR` to a positive Adam learning rate for PyTorch. 

`params` can restart a fit or initialize a higher-rank fit. Set `init='no'` when the supplied parameters should be used without reinitialization. 

## Initialization

The initializers most relevant to the paper are:

- `init='++'` for diametrical K-means++ initialization.
- `init='dc'` for diametrical clustering.
- `init='dc_seg'` to additionally fit a one-component model within every diametrical segment before fitting the mixture.
- `init='ls'` for a least-squares partition.
- `init='ls_seg'` to additionally fit a one-component Normal within every least-squares segment.
- `init='uniform'` for uniform random parameters and equal mixture weights.
- `init='isotropic'` for fixed-norm isotropic random parameter factors (Gaussian-like).
- `init='no'` to retain supplied `params`, for example when increasing rank or converting a mixture to an HMM.

The clustering functions themselves use `init='++'` by default. 

## Hidden Markov models

HMM estimation is available through the PyTorch models with `HMM=True`. `samples_per_sequence` tells the HMM where independent time series begin and end:

- `0` or `None` treats all `n` observations as one sequence.
- A positive integer gives a common sequence length; `n` must be divisible by it. E.g., `1200` for HCP resting-state data.
- A list gives explicit sequence lengths. A listed pattern may repeat when its sum divides `n`.

For example, 12 subjects with 1,200 samples each can be specified as `samples_per_sequence=1200`. For unequal lengths, use a list such as `[900, 1200, 1050]`. Transitions are never counted across sequence boundaries.

With the high-level interface:

```python
hmm_options = {
    'modelname': 'Complex_ACG',
    'rank': 5,
    'LR': 0.05,
    'HMM': True,
    'init': 'dc',
}

hmm_params, states, loglik = train_model(data_train=X, K=3, options=hmm_options, params=mixture_params, samples_per_sequence=1200)
```

The HMM return is a one-hot encoding of the Viterbi state path (as opposed to the posterior for mixtures). To significantly ease fitting of the HMM, start by fitting a mixture model without HMM and use the parameters as input to the HMM with `init='no'`. The transition matrix and delta-parameter of the HMM will automatically be initialized. 

## Complete options reference for the high-level implementation

The high-level `options` dictionary accepts:

- `modelname` (required): for the paper methods, use `'ACG'`, `'Complex_ACG'`, `'Normal'`, `'Complex_Normal'`, `'diametrical'`, or `'complex_diametrical'`. Names belonging to work-in-progress models are listed [below](#additional-models-work-in-progress).
- `rank` (required for some models like ACG and Normal): an integer from `1` to `p` for a factorized model, or `'fullrank'` for the full-covariance NumPy implementation. In the direct NumPy constructors, `None` and `0` also select full covariance.
- `init` (required): one of the applicable values in [Initialization](#initialization).
- `LR=0`: `0` selects NumPy/EM; a positive value selects PyTorch and is used as the Adam learning rate. 
- `HMM=False`: enable a PyTorch HMM. See [Hidden Markov models](#hidden-markov-models).
- `tol=1e-10`: smallest relative objective improvement considered significant.
- `max_iter=100000`: maximum number of EM, Adam, or clustering iterations.
- `num_repl=1`: number of independent initializations; the fit with the best objective is returned.
- `threads=8`: number of PyTorch CPU threads.
- `decrease_lr_on_plateau=False`: reduce the PyTorch learning rate once by a factor of ten after the first plateau.
- `num_comparison=50`: number of iterations without an improvement larger than `tol` before declaring a plateau.
- `force_gamma_same=False`: share the isotropic noise parameter `gamma` across components in low-rank Normal, complex Normal, Singular Wishart, and Wrapped Normal models.

`train_model` separately accepts `params`, `suppress_output`, and `samples_per_sequence`; the last is defined in the [HMM section](#hidden-markov-models). Clustering branches use only the settings relevant to clustering and ignore mixture-specific values.

## Returned values and parameters

`posterior` has shape `(K, n)`. Mixture entries are posterior component probabilities; clustering and HMM entries are one-hot assignments. `loglik` is the retained objective history. For clustering, the high-level parameter dictionary is `{'C': C}`.

Restartable mixture dictionaries contain:

- ACG: `M` and `pi` for a factorized fit, or `Psi` and `pi` for a full-covariance NumPy fit.
- Normal: `M`, `gamma`, and `pi` for a factorized fit, or `Psi` and `pi` for a full-covariance NumPy fit.
- Watson: `mu`, `kappa`, and `pi`.
- PyTorch HMMs: the same emission parameters plus transition matrix `T`; `pi` is the initial-state distribution.

Here `M` has shape `(K, p, rank)`, `Psi` has shape `(K, p, p)`, `pi` has shape `(K,)`, and `T` has shape `(K, K)`.

## Additional models: work in progress

The following models are not described in the PNAS-paper and are part of ongoing distribution extensions to the repo.

### Frame representations

`PCMM.phase_coherence_kmeans` provides `grassmann_clustering` for orthonormal arrays of shape `(n, p, q)` and `weighted_grassmann_clustering` for scaled frames of the same shape. `PCMM.PCMMnumpy` and `PCMM.PCMMtorch` provide MACG models for orthonormal frames and Singular Wishart models for scaled frames. For scaled observations, the sum of squared column norms must equal `p`; the high-level helper currently fixes `q=2`, while direct constructors accept another `q`.

Use `init='gc'` or `'gc_seg'` for MACG and `init='wgc'` or `'wgc_seg'` for Singular Wishart. Their K-means++ versions use `init='gc++'` and `init='wgc++'`.

### Torus representations

`PCMM.phase_coherence_kmeans` provides `torus_clustering` and `quotient_torus_clustering` for arrays of angles `(n, p)` or componentwise nonzero complex phasors. The quotient version removes a common global phase. `PCMM.PCMMtorch.WrappedNormal` provides a PyTorch wrapped Normal mixture; use `init='tc'` or `'qtc'` for torus or quotient-torus initialization.

### Other distributions

`PCMM.PCMMtorch` also contains real and complex Bingham models. [`PCMM/PCMMtorchAdditional.py`](PCMM/PCMMtorchAdditional.py) contains experimental von Mises-Fisher, Fisher-Bingham, matrix Fisher, matrix Bingham, and matrix Fisher-Bingham models. [`PCMM/VMVM_PCMMtorch.py`](PCMM/VMVM_PCMMtorch.py) contains an experimental multivariate von Mises model. These interfaces and normalizing-constant approximations may change.

The high-level names currently include `'Bingham'`, `'Complex_Bingham'`, `'MACG'`, `'SingularWishart'`, `'WrappedNormal'`, `'VMVM'`, `'grassmann'`, and `'weighted_grassmann'`. Bingham, Wrapped Normal, and VMVM require `LR>0`.

## Repository structure

- [`PCMM/`](PCMM/): model implementations, optimization loops, clustering algorithms, and helpers.
- [`paper/`](paper/): data preparation, experiments, and figure code specific to the paper; see its short [README](paper/README.md) for the separate environment requirements.
- [`tutorial.ipynb`](tutorial.ipynb): an executable introduction from analytic signals to clustering and mixture fits.

## Citing the work

If you use PCMM, please cite:

**Uncovering dynamic human brain phase coherence networks**  
Anders S. Olsen, Anders Brammer, Patrick M. Fisher, Morten Mørup.  
[https://www.pnas.org/doi/10.1073/pnas.2518287123](https://www.pnas.org/doi/10.1073/pnas.2518287123)
