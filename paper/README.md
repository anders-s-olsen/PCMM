# Paper code

This folder contains the data preparation, experiments, and figure code for *Uncovering dynamic human brain phase coherence networks*. The main PCMM interface and its minimal installation are documented in the [top-level README](../readme2.md); this folder requires the broader analysis environment used for the paper.

```bash
conda install ipykernel pandas h5py matplotlib seaborn scikit-learn joblib nibabel nilearn mne
pip install tqdm networkx colorcet
```

PyTorch is required for the gradient-based fits, and MATLAB is required only for the included `.m` preprocessing scripts. Data are not included here.

## Contents

- **`data/`** contains the fMRI inputs and derived files used by the analyses, including subject splits, task-onset metadata, saved results, and preprocessing scripts. The preprocessing code covers filtering, nuisance regression, analytic-signal phase and amplitude extraction, phase representations, and HDF5 export.

- **`experiments_realdata/`** contains HCP resting-state and task-fMRI model-order and rank experiments, result aggregation and analysis notebooks, supervised comparisons, and HPC submission templates.

- **`experiments_phaserando/`** constructs phase-controlled and phase-randomized data, fits the clustering and mixture models, and analyzes initialization and phase-control experiments.

- **`synthetic_analysis/`** contains low-dimensional geometry, noise, and anisotropy simulations and the notebooks and MATLAB scripts used to make the corresponding figures.

- **`atlas_figure/`** contains the notebook and surface or mask assets used to visualize the Schaefer atlas.

The files at the top of this folder provide shared paper-specific helpers, task-volume extraction, consistency checks, and a weighted-Grassmann walkthrough.
