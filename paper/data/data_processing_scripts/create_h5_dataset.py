import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
subjects = np.loadtxt('paper/data/255unrelatedsubjectsIDs.txt', dtype='str')

# ################### EEG, complex
num_subjects = 1
subject_split = 20
complex_eigenvectors_train = []
complex_eigenvectors_test = []
num_pts = []

for i in range(num_subjects):
    print(i)
    data_angle = np.loadtxt('paper/data/processed/EEG/EPCTL{:02d}_angle.csv'.format(i+1), delimiter=',')
    data_complex = np.exp(1j*data_angle)
    data_complex /= np.linalg.norm(data_complex, axis=1, keepdims=True)
    num_pts.append(data_complex.shape[0])
    if i < subject_split:
        complex_eigenvectors_train.append(data_complex)
    else:
        complex_eigenvectors_test.append(data_complex)
    plt.figure()
    plt.imshow(np.abs(data_complex.T.conj()@data_complex), aspect='auto')
    plt.colorbar()
    plt.savefig('paper/data/data_processing_scripts/eeg_cov_matrices/tmp'+str(i+1)+'.png')
    plt.close()

complex_eigenvectors_train = np.concatenate(complex_eigenvectors_train)
complex_eigenvectors_test = np.concatenate(complex_eigenvectors_test)
np.savetxt('paper/data/processed/EEG/num_pts.csv', num_pts, delimiter=',')

with h5.File('paper/data/processed/EEG.h5','w') as f:
    f.create_dataset('U_complex_train',data=complex_eigenvectors_train)
    f.create_dataset('U_complex_test',data=complex_eigenvectors_test)


# ################### MOTOR-SOCIAL, GSR, complex
# add_GSR = '_GSR'
add_GSR = ''
task1 = 'MOTOR'
task2 = 'SOCIAL'
num_points = np.array([284,274])
subject_split = 155
sum_num_points = np.sum(num_points)
num_subs = len(subjects)
complex_eigenvectors_train = np.zeros((sum_num_points//2*subject_split,116,1),dtype='complex')
complex_eigenvectors_test1 = np.zeros((sum_num_points//2*subject_split,116,1),dtype='complex')
complex_eigenvectors_test2 = np.zeros((sum_num_points*(num_subs-subject_split),116,1),dtype='complex')
hilbert_amplitude_train = np.zeros((sum_num_points//2*subject_split,116))
hilbert_amplitude_test1 = np.zeros((sum_num_points//2*subject_split,116))
hilbert_amplitude_test2 = np.zeros((sum_num_points*(num_subs-subject_split),116))
cos_eigenvectors_train = np.zeros((sum_num_points//2*subject_split,116,2))
cos_eigenvectors_test1 = np.zeros((sum_num_points//2*subject_split,116,2))
cos_eigenvectors_test2 = np.zeros((sum_num_points*(num_subs-subject_split),116,2))
cos_eigenvalues_train = np.zeros((sum_num_points//2*subject_split,2))
cos_eigenvalues_test1 = np.zeros((sum_num_points//2*subject_split,2))
cos_eigenvalues_test2 = np.zeros((sum_num_points*(num_subs-subject_split),2))
timeseries_train = np.zeros((sum_num_points//2*subject_split,116,1))
timeseries_test1 = np.zeros((sum_num_points//2*subject_split,116,1))
timeseries_test2 = np.zeros((sum_num_points*(num_subs-subject_split),116,1))
#shuffle subjects
np.random.seed(0)
np.random.shuffle(subjects)
for i,sub in enumerate(subjects):
    print(i)
    complex_evecs_train = []
    complex_evecs_test = []
    amplitude_train = []
    amplitude_test = []
    cos_evecs_train = []
    cos_evecs_test = []
    cos_evals_train = []
    cos_evals_test = []
    ts_train = []
    ts_test = []
    for j,task in enumerate([task1,task2]):
        evecs_real = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_Atlas_real.csv',delimiter=',')
        evecs_imag = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_Atlas_imag.csv',delimiter=',')
        evecs_complex = evecs_real + 1j*evecs_imag
        hilbert_amplitude = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_Atlas_amplitude.csv',delimiter=',')
        tmp_evecs = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_Atlas.csv',delimiter=',')
        evecs_cos = np.zeros((num_points[j],116,2))
        evecs_cos[:,:,0] = tmp_evecs[::2,:]
        evecs_cos[:,:,1] = tmp_evecs[1::2,:]
        tmp_evals = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_Atlas_evs.csv',delimiter=',')
        evals_cos = np.zeros((num_points[j],2))
        evals_cos[:,0] = tmp_evals[::2]
        evals_cos[:,1] = tmp_evals[1::2]
        timeseries = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_Atlas_timeseries.csv',delimiter=',')
        # choose one or two as train
        if i<subject_split: #split each scan into train and test
            block = np.random.choice([1,2])
            if block == 1:
                complex_evecs_train.append(evecs_complex[:(num_points[j]//2)])
                complex_evecs_test.append(evecs_complex[(num_points[j]//2):])
                amplitude_train.append(hilbert_amplitude[:(num_points[j]//2)])
                amplitude_test.append(hilbert_amplitude[(num_points[j]//2):])
                cos_evecs_train.append(evecs_cos[:(num_points[j]//2)])
                cos_evecs_test.append(evecs_cos[(num_points[j]//2):])
                cos_evals_train.append(evals_cos[:(num_points[j]//2)])
                cos_evals_test.append(evals_cos[(num_points[j]//2):])
                ts_train.append(timeseries[:(num_points[j]//2)])
                ts_test.append(timeseries[(num_points[j]//2):])
            else:
                complex_evecs_train.append(evecs_complex[(num_points[j]//2):])
                complex_evecs_test.append(evecs_complex[:(num_points[j]//2)])
                amplitude_train.append(hilbert_amplitude[(num_points[j]//2):])
                amplitude_test.append(hilbert_amplitude[:(num_points[j]//2)])
                cos_evecs_train.append(evecs_cos[(num_points[j]//2):])
                cos_evecs_test.append(evecs_cos[:(num_points[j]//2)])
                cos_evals_train.append(evals_cos[(num_points[j]//2):])
                cos_evals_test.append(evals_cos[:(num_points[j]//2)])
                ts_train.append(timeseries[(num_points[j]//2):])
                ts_test.append(timeseries[:(num_points[j]//2)])
        else: #make only test set
            complex_evecs_test.append(evecs_complex)
            amplitude_test.append(hilbert_amplitude)
            cos_evecs_test.append(evecs_cos)
            cos_evals_test.append(evals_cos)
            ts_test.append(timeseries)
    
    #concatenate over tasks
    if i<subject_split:
        complex_evecs_train = np.concatenate(complex_evecs_train)
        amplitude_train = np.concatenate(amplitude_train)
        cos_evecs_train = np.concatenate(cos_evecs_train)
        cos_evals_train = np.concatenate(cos_evals_train)
        ts_train = np.concatenate(ts_train)
    complex_evecs_test = np.concatenate(complex_evecs_test)
    amplitude_test = np.concatenate(amplitude_test)
    cos_evecs_test = np.concatenate(cos_evecs_test)
    cos_evals_test = np.concatenate(cos_evals_test)
    ts_test = np.concatenate(ts_test)
    
    if i<subject_split:
        complex_eigenvectors_train[sum_num_points//2*i:sum_num_points//2*(i+1),:,0] = complex_evecs_train
        complex_eigenvectors_test1[sum_num_points//2*i:sum_num_points//2*(i+1),:,0] = complex_evecs_test
        hilbert_amplitude_train[sum_num_points//2*i:sum_num_points//2*(i+1),:] = amplitude_train
        hilbert_amplitude_test1[sum_num_points//2*i:sum_num_points//2*(i+1),:] = amplitude_test
        cos_eigenvectors_train[sum_num_points//2*i:sum_num_points//2*(i+1),:,:] = cos_evecs_train
        cos_eigenvectors_test1[sum_num_points//2*i:sum_num_points//2*(i+1),:,:] = cos_evecs_test
        cos_eigenvalues_train[sum_num_points//2*i:sum_num_points//2*(i+1),:] = cos_evals_train
        cos_eigenvalues_test1[sum_num_points//2*i:sum_num_points//2*(i+1),:] = cos_evals_test
        timeseries_train[sum_num_points//2*i:sum_num_points//2*(i+1),:,0] = ts_train
        timeseries_test1[sum_num_points//2*i:sum_num_points//2*(i+1),:,0] = ts_test
    else:
        complex_eigenvectors_test2[sum_num_points*(i-subject_split):sum_num_points*(i-subject_split+1),:,0] = complex_evecs_test
        hilbert_amplitude_test2[sum_num_points*(i-subject_split):sum_num_points*(i-subject_split+1),:] = amplitude_test
        cos_eigenvectors_test2[sum_num_points*(i-subject_split):sum_num_points*(i-subject_split+1),:,:] = cos_evecs_test
        cos_eigenvalues_test2[sum_num_points*(i-subject_split):sum_num_points*(i-subject_split+1),:] = cos_evals_test
        timeseries_test2[sum_num_points*(i-subject_split):sum_num_points*(i-subject_split+1),:,0] = ts_test
    
with h5.File('paper/data/processed/'+task1+task2+'fMRI_SchaeferTian116'+add_GSR+'.h5','w') as f:
    f.create_dataset('U_complex_train',data=complex_eigenvectors_train)
    f.create_dataset('U_complex_test1',data=complex_eigenvectors_test1)
    f.create_dataset('U_complex_test2',data=complex_eigenvectors_test2)
    f.create_dataset('A_train',data=hilbert_amplitude_train)
    f.create_dataset('A_test1',data=hilbert_amplitude_test1)
    f.create_dataset('A_test2',data=hilbert_amplitude_test2)
    f.create_dataset('U_cos_train',data=cos_eigenvectors_train)
    f.create_dataset('U_cos_test1',data=cos_eigenvectors_test1)
    f.create_dataset('U_cos_test2',data=cos_eigenvectors_test2)
    f.create_dataset('L_cos_train',data=cos_eigenvalues_train)
    f.create_dataset('L_cos_test1',data=cos_eigenvalues_test1)
    f.create_dataset('L_cos_test2',data=cos_eigenvalues_test2)
    f.create_dataset('timeseries_train',data=timeseries_train)
    f.create_dataset('timeseries_test1',data=timeseries_test1)
    f.create_dataset('timeseries_test2',data=timeseries_test2)


# # ################### REST, GSR, complex
# # add_GSR = '_GSR'
# add_GSR = ''
# task1 = 'REST1'
# task2 = 'REST2'
# num_points = np.array([1200,1200])
# subject_split = 155
# sum_num_points = np.sum(num_points)
# num_subs = len(subjects)
# complex_eigenvectors_train = np.zeros((sum_num_points//2*subject_split,116,1),dtype='complex')
# complex_eigenvectors_test1 = np.zeros((sum_num_points//2*subject_split,116,1),dtype='complex')
# complex_eigenvectors_test2 = np.zeros((sum_num_points*(num_subs-subject_split),116,1),dtype='complex')
# hilbert_amplitude_train = np.zeros((sum_num_points//2*subject_split,116))
# hilbert_amplitude_test1 = np.zeros((sum_num_points//2*subject_split,116))
# hilbert_amplitude_test2 = np.zeros((sum_num_points*(num_subs-subject_split),116))
# cos_eigenvectors_train = np.zeros((sum_num_points//2*subject_split,116,2))
# cos_eigenvectors_test1 = np.zeros((sum_num_points//2*subject_split,116,2))
# cos_eigenvectors_test2 = np.zeros((sum_num_points*(num_subs-subject_split),116,2))
# cos_eigenvalues_train = np.zeros((sum_num_points//2*subject_split,2))
# cos_eigenvalues_test1 = np.zeros((sum_num_points//2*subject_split,2))
# cos_eigenvalues_test2 = np.zeros((sum_num_points*(num_subs-subject_split),2))
# timeseries_train = np.zeros((sum_num_points//2*subject_split,116,1))
# timeseries_test1 = np.zeros((sum_num_points//2*subject_split,116,1))
# timeseries_test2 = np.zeros((sum_num_points*(num_subs-subject_split),116,1))
# #shuffle subjects
# np.random.seed(0)
# np.random.shuffle(subjects)
# for i,sub in enumerate(subjects):
#     print(i)
#     complex_evecs_train = []
#     complex_evecs_test = []
#     amplitude_train = []
#     amplitude_test = []
#     cos_evecs_train = []
#     cos_evecs_test = []
#     cos_evals_train = []
#     cos_evals_test = []
#     ts_train = []
#     ts_test = []
#     if i<subject_split: #split each scan into train and test
#         block = np.random.permutation([1,2])
#     for j,task in enumerate([task1,task2]):
#         evecs_real = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_Atlas_MSMAll_hp2000_clean_real.csv',delimiter=',')
#         evecs_imag = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_Atlas_MSMAll_hp2000_clean_imag.csv',delimiter=',')
#         evecs_complex = evecs_real + 1j*evecs_imag
#         hilbert_amplitude = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_Atlas_MSMAll_hp2000_clean_amplitude.csv',delimiter=',')
#         tmp_evecs = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_Atlas_MSMAll_hp2000_clean.csv',delimiter=',')
#         evecs_cos = np.zeros((num_points[j],116,2))
#         evecs_cos[:,:,0] = tmp_evecs[::2,:]
#         evecs_cos[:,:,1] = tmp_evecs[1::2,:]
#         tmp_evals = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_Atlas_MSMAll_hp2000_clean_evs.csv',delimiter=',')
#         evals_cos = np.zeros((num_points[j],2))
#         evals_cos[:,0] = tmp_evals[::2]
#         evals_cos[:,1] = tmp_evals[1::2]
#         timeseries = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_Atlas_MSMAll_hp2000_clean_timeseries.csv',delimiter=',')
#         # choose one or two as train
#         if i<subject_split: #split each scan into train and test
#             if block[j] == 1:
#                 complex_evecs_train.append(evecs_complex)
#                 amplitude_train.append(hilbert_amplitude)
#                 cos_evecs_train.append(evecs_cos)
#                 cos_evals_train.append(evals_cos)
#                 ts_train.append(timeseries)
#             else:
#                 complex_evecs_test.append(evecs_complex)
#                 amplitude_test.append(hilbert_amplitude)
#                 cos_evecs_test.append(evecs_cos)
#                 cos_evals_test.append(evals_cos)
#                 ts_test.append(timeseries)
#         else: #make only test set
#             complex_evecs_test.append(evecs_complex)
#             amplitude_test.append(hilbert_amplitude)
#             cos_evecs_test.append(evecs_cos)
#             cos_evals_test.append(evals_cos)
#             ts_test.append(timeseries)
    
#     #concatenate over tasks
#     if i<subject_split:
#         complex_evecs_train = np.concatenate(complex_evecs_train)
#         amplitude_train = np.concatenate(amplitude_train)
#         cos_evecs_train = np.concatenate(cos_evecs_train)
#         cos_evals_train = np.concatenate(cos_evals_train)
#         ts_train = np.concatenate(ts_train)
#     complex_evecs_test = np.concatenate(complex_evecs_test)
#     amplitude_test = np.concatenate(amplitude_test)
#     cos_evecs_test = np.concatenate(cos_evecs_test)
#     cos_evals_test = np.concatenate(cos_evals_test)
#     ts_test = np.concatenate(ts_test)
    
#     if i<subject_split:
#         complex_eigenvectors_train[sum_num_points//2*i:sum_num_points//2*(i+1),:,0] = complex_evecs_train
#         complex_eigenvectors_test1[sum_num_points//2*i:sum_num_points//2*(i+1),:,0] = complex_evecs_test
#         hilbert_amplitude_train[sum_num_points//2*i:sum_num_points//2*(i+1),:] = amplitude_train
#         hilbert_amplitude_test1[sum_num_points//2*i:sum_num_points//2*(i+1),:] = amplitude_test
#         cos_eigenvectors_train[sum_num_points//2*i:sum_num_points//2*(i+1),:,:] = cos_evecs_train
#         cos_eigenvectors_test1[sum_num_points//2*i:sum_num_points//2*(i+1),:,:] = cos_evecs_test
#         cos_eigenvalues_train[sum_num_points//2*i:sum_num_points//2*(i+1),:] = cos_evals_train
#         cos_eigenvalues_test1[sum_num_points//2*i:sum_num_points//2*(i+1),:] = cos_evals_test
#         timeseries_train[sum_num_points//2*i:sum_num_points//2*(i+1),:,0] = ts_train
#         timeseries_test1[sum_num_points//2*i:sum_num_points//2*(i+1),:,0] = ts_test
#     else:
#         complex_eigenvectors_test2[sum_num_points*(i-subject_split):sum_num_points*(i-subject_split+1),:,0] = complex_evecs_test
#         hilbert_amplitude_test2[sum_num_points*(i-subject_split):sum_num_points*(i-subject_split+1),:] = amplitude_test
#         cos_eigenvectors_test2[sum_num_points*(i-subject_split):sum_num_points*(i-subject_split+1),:,:] = cos_evecs_test
#         cos_eigenvalues_test2[sum_num_points*(i-subject_split):sum_num_points*(i-subject_split+1),:] = cos_evals_test
#         timeseries_test2[sum_num_points*(i-subject_split):sum_num_points*(i-subject_split+1),:,0] = ts_test
    
# with h5.File('paper/data/processed/'+task1+task2+'fMRI_SchaeferTian116'+add_GSR+'.h5','w') as f:
#     f.create_dataset('U_complex_train',data=complex_eigenvectors_train)
#     f.create_dataset('U_complex_test1',data=complex_eigenvectors_test1)
#     f.create_dataset('U_complex_test2',data=complex_eigenvectors_test2)
#     f.create_dataset('A_train',data=hilbert_amplitude_train)
#     f.create_dataset('A_test1',data=hilbert_amplitude_test1)
#     f.create_dataset('A_test2',data=hilbert_amplitude_test2)
#     f.create_dataset('U_cos_train',data=cos_eigenvectors_train)
#     f.create_dataset('U_cos_test1',data=cos_eigenvectors_test1)
#     f.create_dataset('U_cos_test2',data=cos_eigenvectors_test2)
#     f.create_dataset('L_cos_train',data=cos_eigenvalues_train)
#     f.create_dataset('L_cos_test1',data=cos_eigenvalues_test1)
#     f.create_dataset('L_cos_test2',data=cos_eigenvalues_test2)
#     f.create_dataset('timeseries_train',data=timeseries_train)
#     f.create_dataset('timeseries_test1',data=timeseries_test1)
#     f.create_dataset('timeseries_test2',data=timeseries_test2)