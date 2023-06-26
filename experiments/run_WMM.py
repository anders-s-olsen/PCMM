import h5py
import numpy as np

# datah5_train = h5py.File('../data/processed/fMRI_atlas_RL1.h5', 'r')
data_train = np.array(h5py.File('data/processed/fMRI_atlas_RL1.h5', 'r')['Dataset']).T
data_test = np.array(h5py.File('data/processed/fMRI_atlas_RL2.h5', 'r')['Dataset']).T

