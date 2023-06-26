import h5py
import numpy as np

datah5_train = h5py.File('../data/processed/fMRI_atlas_RL1.h5', 'r')
datah5_test = h5py.File('../data/processed/fMRI_atlas_RL1.h5', 'r')

data_train = np.zeros((29,120,num_regions))
data_test = np.zeros((29,120,num_regions))
for idx,subject in enumerate(list(datah5.keys())):
    data_train[idx] = torch.DoubleTensor(np.array(datah5[subject][0:120]))
    data_test[idx] = torch.DoubleTensor(np.array(datah5[subject][120:]))
data_test_concat = torch.concatenate([data_test[sub] for sub in range(data_test.shape[0])])