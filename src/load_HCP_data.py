import numpy as np
import h5py as h5
from tqdm import tqdm


def load_real_data(options,folder,subjectlist,suppress_output=False):
    try:
        subjectlist = subjectlist[:options['num_subjects']]
    except:
        pass
    if folder=='fMRI_SchaeferTian116_GSR' or folder=='fMRI_SchaeferTian116':
        tt = 1
        num_rois = 116
    elif folder=='fMRI_full_GSR' or folder=='fMRI_full':
        tt = 2
        num_rois = 91282
    else:
        raise ValueError('Invalid folder')

    data_train_all = np.zeros((1200*len(subjectlist),num_rois,2))
    L_train_all = np.zeros((1200*len(subjectlist),2))
    data_test_all = np.zeros((1200*len(subjectlist),num_rois,2))
    L_test_all = np.zeros((1200*len(subjectlist),2))

    for s,subject in tqdm(enumerate(subjectlist),disable=suppress_output):
        if tt==1:
            data1 = np.loadtxt('data/processed/'+folder+'/'+str(subject)+'_rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.csv',delimiter=',')
            data2 = np.loadtxt('data/processed/'+folder+'/'+str(subject)+'_rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.csv',delimiter=',')
        elif tt==2:
            file1 = 'data/processed/'+folder+'/'+str(subject)+'_rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.h5'
            file2 = 'data/processed/'+folder+'/'+str(subject)+'_rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.h5'
            with h5.File(file1, 'r') as f:
                data1 = f['data'][:].T
            with h5.File(file2, 'r') as f:
                data2 = f['data'][:].T
        else:
            raise ValueError('Invalid folder')
        if num_eigs==1:
            data_train = data1[::2,:]
            data_test = data2[::2,:]
        elif num_eigs == 2:
            p = data1.shape[1]
            data_train = np.zeros((data1.shape[0]//2,p,2))
            data_train[:,:,0] = data1[::2,:]
            data_train[:,:,1] = data1[1::2,:]
            data_test = np.zeros((data2.shape[0]//2,p,2))
            data_test[:,:,0] = data2[::2,:]
            data_test[:,:,1] = data2[1::2,:]
        data_train_all[1200*s:1200*(s+1)] = data_train
        data_test_all[1200*s:1200*(s+1)] = data_test

    # if options['LR']==0:
    #     data_train_all = np.ascontiguousarray(data_train_all)
    #     data_test_all = np.ascontiguousarray(data_test_all)

    return data_train_all,data_test_all