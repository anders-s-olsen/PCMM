import h5py
import numpy as np

# datah5_train = h5py.File('../data/processed/fMRI_atlas_RL1.h5', 'r')
datah5_train = h5py.File('data/processed/fMRI_atlas_RL1.h5', 'r')
datah5_test = h5py.File('data/processed/fMRI_atlas_RL2.h5', 'r')

data_train = np.zeros((2537224,454))

for idx,subject in enumerate(list(datah5_train.keys())):
    length = datah5[]
    data_train[idx] = torch.DoubleTensor(np.array(datah5[subject][0:120]))
data_test_concat = torch.concatenate([data_test[sub] for sub in range(data_test.shape[0])])