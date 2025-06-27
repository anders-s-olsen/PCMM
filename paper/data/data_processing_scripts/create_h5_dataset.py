import numpy as np
import h5py as h5
# import matplotlib.pyplot as plt
subjects = np.loadtxt('paper/data/255unrelatedsubjectsIDs.txt', dtype='str')

# # ################### All tasks
# # add_GSR = '_GSR'
# add_GSR = ''
# tasks = ['EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM']
# # tasks = ['EMOTION','GAMBLING','LANGUAGE','MOTOR']
# subject_split = 155
# num_subs = len(subjects)
# num_points = np.array([176,253,316,284,232,274,405])
# np.random.seed(0)
# np.random.shuffle(subjects)
# subjects = subjects[subjects!='134627']

# for j,task in enumerate(tasks):
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
#     for i,sub in enumerate(subjects):
#         print(sub, task)
#         evecs_real = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_Atlas_real.csv',delimiter=',')
#         if evecs_real.shape[0] != num_points[j]:
#             raise ValueError(f"Number of points for subject {sub} in task {task} does not match expected {num_points[j]}. Found {evecs_real.shape[0]}.")
#         evecs_imag = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_Atlas_imag.csv',delimiter=',')
#         evecs_complex = evecs_real + 1j*evecs_imag
#         hilbert_amplitude = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_Atlas_amplitude.csv',delimiter=',')
#         tmp_evecs = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_Atlas.csv',delimiter=',')
#         evecs_cos = np.zeros((num_points[j],116,2))
#         evecs_cos[:,:,0] = tmp_evecs[::2,:]
#         evecs_cos[:,:,1] = tmp_evecs[1::2,:]
#         tmp_evals = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_Atlas_evs.csv',delimiter=',')
#         evals_cos = np.zeros((num_points[j],2))
#         evals_cos[:,0] = tmp_evals[::2]
#         evals_cos[:,1] = tmp_evals[1::2]
#         timeseries = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_Atlas_timeseries.csv',delimiter=',')
#         # choose one or two as train
#         if i<subject_split: #split each scan into train and test
#             complex_evecs_train.append(evecs_complex)
#             amplitude_train.append(hilbert_amplitude)
#             cos_evecs_train.append(evecs_cos)
#             cos_evals_train.append(evals_cos)
#             ts_train.append(timeseries)
#         else: #make only test set
#             complex_evecs_test.append(evecs_complex)
#             amplitude_test.append(hilbert_amplitude)
#             cos_evecs_test.append(evecs_cos)
#             cos_evals_test.append(evals_cos)
#             ts_test.append(timeseries)    #concatenate over tasks
#     complex_evecs_train = np.concatenate(complex_evecs_train)
#     amplitude_train = np.concatenate(amplitude_train)
#     cos_evecs_train = np.concatenate(cos_evecs_train)
#     cos_evals_train = np.concatenate(cos_evals_train)
#     ts_train = np.concatenate(ts_train)
#     complex_evecs_test = np.concatenate(complex_evecs_test)
#     amplitude_test = np.concatenate(amplitude_test)
#     cos_evecs_test = np.concatenate(cos_evecs_test)
#     cos_evals_test = np.concatenate(cos_evals_test)
#     ts_test = np.concatenate(ts_test)
#     with h5.File('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'.h5','w') as f:
#         f.create_dataset('U_complex_train',data=complex_evecs_train)
#         f.create_dataset('U_complex_test',data=complex_evecs_test)
#         f.create_dataset('A_train',data=amplitude_train)
#         f.create_dataset('A_test',data=amplitude_test)
#         f.create_dataset('U_cos_train',data=cos_evecs_train)
#         f.create_dataset('U_cos_test',data=cos_evecs_test)
#         f.create_dataset('L_cos_train',data=cos_evals_train)
#         f.create_dataset('L_cos_test',data=cos_evals_test)
#         f.create_dataset('timeseries_train',data=ts_train)
#         f.create_dataset('timeseries_test',data=ts_test)

# tasks = ['EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM']
# # concatenate the seven task h5 files into one file
# with h5.File('paper/data/processed/all_tasksfMRI_SchaeferTian116'+add_GSR+'.h5','w') as f:
#     U_complex_train = []
#     U_complex_test = []
#     A_train = []
#     A_test = []
#     U_cos_train = []
#     U_cos_test = []
#     L_cos_train = []
#     L_cos_test = []
#     timeseries_train = []
#     timeseries_test = []
#     for i in range(len(subjects)):
#         print(i)
#         for j,task in enumerate(tasks):
#             with h5.File('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'.h5','r') as f_task:
#                 if i < 155:  # subject_split
#                     U_complex_train.append(f_task['U_complex_train'][:][i*num_points[j]:(i+1)*num_points[j]])
#                     A_train.append(f_task['A_train'][:][i*num_points[j]:(i+1)*num_points[j]])
#                     U_cos_train.append(f_task['U_cos_train'][:][i*num_points[j]:(i+1)*num_points[j]])
#                     L_cos_train.append(f_task['L_cos_train'][:][i*num_points[j]:(i+1)*num_points[j]])
#                     timeseries_train.append(f_task['timeseries_train'][:][i*num_points[j]:(i+1)*num_points[j]])
#                 if i < 99:
#                     U_complex_test.append(f_task['U_complex_test'][:][i*num_points[j]:(i+1)*num_points[j]])
#                     A_test.append(f_task['A_test'][:][i*num_points[j]:(i+1)*num_points[j]])
#                     U_cos_test.append(f_task['U_cos_test'][:][i*num_points[j]:(i+1)*num_points[j]])
#                     L_cos_test.append(f_task['L_cos_test'][:][i*num_points[j]:(i+1)*num_points[j]])
#                     timeseries_test.append(f_task['timeseries_test'][:][i*num_points[j]:(i+1)*num_points[j]])

#     U_complex_train = np.concatenate(U_complex_train)
#     U_complex_test = np.concatenate(U_complex_test)
#     A_train = np.concatenate(A_train)
#     A_test = np.concatenate(A_test)
#     U_cos_train = np.concatenate(U_cos_train)
#     U_cos_test = np.concatenate(U_cos_test)
#     L_cos_train = np.concatenate(L_cos_train)
#     L_cos_test = np.concatenate(L_cos_test)
#     timeseries_train = np.concatenate(timeseries_train)
#     timeseries_test = np.concatenate(timeseries_test)
#     f.create_dataset('U_complex_train',data=U_complex_train)
#     f.create_dataset('U_complex_test',data=U_complex_test)
#     f.create_dataset('A_train',data=A_train)
#     f.create_dataset('A_test',data=A_test)
#     f.create_dataset('U_cos_train',data=U_cos_train)
#     f.create_dataset('U_cos_test',data=U_cos_test)
#     f.create_dataset('L_cos_train',data=L_cos_train)
#     f.create_dataset('L_cos_test',data=L_cos_test)
#     f.create_dataset('timeseries_train',data=timeseries_train)
#     f.create_dataset('timeseries_test',data=timeseries_test)
    

# ################### REST
# add_GSR = '_GSR'
add_GSR = ''
task1 = 'REST1'
task2 = 'REST2'
num_points = np.array([1200,1200])
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
    if i<subject_split: #split each scan into train and test
        block = np.random.permutation([1,2])
    for j,task in enumerate([task1,task2]):
        evecs_real = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_Atlas_MSMAll_hp2000_clean_real.csv',delimiter=',')
        evecs_imag = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_Atlas_MSMAll_hp2000_clean_imag.csv',delimiter=',')
        evecs_complex = evecs_real + 1j*evecs_imag
        hilbert_amplitude = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_Atlas_MSMAll_hp2000_clean_amplitude.csv',delimiter=',')
        tmp_evecs = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_Atlas_MSMAll_hp2000_clean.csv',delimiter=',')
        evecs_cos = np.zeros((num_points[j],116,2))
        evecs_cos[:,:,0] = tmp_evecs[::2,:]
        evecs_cos[:,:,1] = tmp_evecs[1::2,:]
        tmp_evals = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_Atlas_MSMAll_hp2000_clean_evs.csv',delimiter=',')
        evals_cos = np.zeros((num_points[j],2))
        evals_cos[:,0] = tmp_evals[::2]
        evals_cos[:,1] = tmp_evals[1::2]
        timeseries = np.loadtxt('paper/data/processed/'+task+'fMRI_SchaeferTian116'+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_Atlas_MSMAll_hp2000_clean_timeseries.csv',delimiter=',')
        # choose one or two as train
        if i<subject_split: #split each scan into train and test
            if block[j] == 1:
                complex_evecs_train.append(evecs_complex)
                amplitude_train.append(hilbert_amplitude)
                cos_evecs_train.append(evecs_cos)
                cos_evals_train.append(evals_cos)
                ts_train.append(timeseries)
            else:
                complex_evecs_test.append(evecs_complex)
                amplitude_test.append(hilbert_amplitude)
                cos_evecs_test.append(evecs_cos)
                cos_evals_test.append(evals_cos)
                ts_test.append(timeseries)
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