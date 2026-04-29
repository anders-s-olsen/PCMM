import numpy as np
import h5py as h5
# import matplotlib.pyplot as plt
subjects = np.loadtxt('paper/data/255unrelatedsubjectsIDs.txt', dtype='str')

# ################### All tasks
add_GSR = '_GSR'
# add_GSR = ''
if add_GSR == '_GSR':
    add_GSR2 = '_GSR'
else:    
    add_GSR2 = '_noGSR'
split_6040_or_8020 = '8020' # '6040' or '8020'
if split_6040_or_8020 == '8020':
    split_idx = 4
atlas = 'SchaeferTian116' # SchaeferTian116, SchaeferTian232, Schaefer400
dataset = 'Atlas_MSMAll' # Atlas_MSMAll, Atlas_MSMAll_hp0_clean_rclean_tclean
if dataset != 'Atlas_MSMAll':
    add_preproc = '2025'
else:
    add_preproc = ''
p = int(atlas[-3:])
tasks = ['EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM']
# tasks = ['EMOTION','GAMBLING','LANGUAGE','MOTOR']
num_subs = len(subjects)
remove_first_vols = np.array([15,11,0,11,11,11,11])
num_points = np.array([176,253,316,284,232,274,405])
num_points_after_removal = num_points - remove_first_vols
sum_points = np.sum(num_points_after_removal)
cumsum_points = np.cumsum(num_points_after_removal) #changed from num_points to num_points_after_removal
cumsum_points = np.insert(cumsum_points,0,0)
np.random.seed(0)
np.random.shuffle(subjects)
subjects = subjects[subjects!='134627'] # reverse the order of removal...
np.savetxt('paper/data/all_tasks_subject_split.txt',subjects,fmt='%s')

if split_6040_or_8020 == '6040':
    num_train_subjects = 155
    num_test_subjects = 99
    train_subjects = subjects[:num_train_subjects]
    test_subjects = subjects[num_train_subjects:]
elif split_6040_or_8020 == '8020':
    num_train_subjects = 203
    num_test_subjects = 51
    test_subjects = subjects[split_idx*num_test_subjects:(split_idx+1)*num_test_subjects]
    if split_idx == 4:
        test_subjects = np.append(test_subjects, subjects[0]) # add back the first subject...
    train_subjects = np.setdiff1d(subjects, test_subjects)

print('Working on dataset:',dataset,'with atlas:',atlas,'and GSR:',add_GSR)
for j,task in enumerate(tasks):
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
    for i,sub in enumerate(subjects):
        print(sub, task)
        evecs_real = np.loadtxt('paper/data/processed/'+task+'fMRI_'+atlas+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_'+dataset+'_real.csv',delimiter=',')
        if evecs_real.shape[0] != num_points[j]:
            raise ValueError(f"Number of points for subject {sub} in task {task} does not match expected {num_points[j]}. Found {evecs_real.shape[0]}.")
        evecs_imag = np.loadtxt('paper/data/processed/'+task+'fMRI_'+atlas+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_'+dataset+'_imag.csv',delimiter=',')
        evecs_complex = evecs_real + 1j*evecs_imag
        hilbert_amplitude = np.loadtxt('paper/data/processed/'+task+'fMRI_'+atlas+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_'+dataset+'_amplitude.csv',delimiter=',')
        tmp_evecs = np.loadtxt('paper/data/processed/'+task+'fMRI_'+atlas+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_'+dataset+'.csv',delimiter=',')
        evecs_cos = np.zeros((num_points[j],p,2))
        evecs_cos[:,:,0] = tmp_evecs[::2,:]
        evecs_cos[:,:,1] = tmp_evecs[1::2,:]
        tmp_evals = np.loadtxt('paper/data/processed/'+task+'fMRI_'+atlas+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_'+dataset+'_evs.csv',delimiter=',')
        evals_cos = np.zeros((num_points[j],2))
        evals_cos[:,0] = tmp_evals[::2]
        evals_cos[:,1] = tmp_evals[1::2]
        timeseries = np.loadtxt('paper/data/processed/'+task+'fMRI_'+atlas+add_GSR+'/'+str(sub)+'_tfMRI_'+task+'_RL_'+dataset+'_timeseries.csv',delimiter=',')
        # choose one or two as train
        # if i<subject_split: #split each scan into train and test
        if sub in train_subjects:
            complex_evecs_train.append(evecs_complex[remove_first_vols[j]:,:])
            amplitude_train.append(hilbert_amplitude[remove_first_vols[j]:,:])
            cos_evecs_train.append(evecs_cos[remove_first_vols[j]:,:])
            cos_evals_train.append(evals_cos[remove_first_vols[j]:,:])
            ts_train.append(timeseries[remove_first_vols[j]:,:])
        elif sub in test_subjects: #make only test set
            complex_evecs_test.append(evecs_complex[remove_first_vols[j]:,:])
            amplitude_test.append(hilbert_amplitude[remove_first_vols[j]:,:])
            cos_evecs_test.append(evecs_cos[remove_first_vols[j]:,:])
            cos_evals_test.append(evals_cos[remove_first_vols[j]:,:])
            ts_test.append(timeseries[remove_first_vols[j]:,:])    #concatenate over tasks
        else:
            raise ValueError(f"Subject {sub} not found in either train or test subjects.")
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
    if split_6040_or_8020 == '6040':
        tmp_filename = 'paper/data/processed/concatenated_datasets/'+add_preproc+task+'fMRI_'+atlas+add_GSR+'.h5'
    elif split_6040_or_8020 == '8020':
        tmp_filename = 'paper/data/processed/concatenated_datasets/'+add_preproc+task+'fMRI_'+atlas+add_GSR+'_8020split'+str(split_idx)+'.h5'
    with h5.File(tmp_filename,'w') as f:
        f.create_dataset('U_complex_train',data=complex_evecs_train)
        f.create_dataset('U_complex_test',data=complex_evecs_test)
        f.create_dataset('A_train',data=amplitude_train)
        f.create_dataset('A_test',data=amplitude_test)
        f.create_dataset('U_cos_train',data=cos_evecs_train)
        f.create_dataset('U_cos_test',data=cos_evecs_test)
        f.create_dataset('L_cos_train',data=cos_evals_train)
        f.create_dataset('L_cos_test',data=cos_evals_test)
        f.create_dataset('timeseries_train',data=ts_train)
        f.create_dataset('timeseries_test',data=ts_test)

tasks = ['EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM']
# concatenate the seven task h5 files into one file
if split_6040_or_8020 == '6040':
    final_filename = 'paper/data/processed/concatenated_datasets/all_tasks'+add_preproc+'fMRI_'+atlas+add_GSR2+'.h5'
elif split_6040_or_8020 == '8020':
    final_filename = 'paper/data/processed/concatenated_datasets/all_tasks'+add_preproc+'fMRI_'+atlas+add_GSR2+'_8020split'+str(split_idx)+'.h5'
with h5.File(final_filename,'w') as f:
    U_complex_train = np.zeros((num_train_subjects*sum_points,p),dtype='complex')
    U_complex_test = np.zeros((num_test_subjects*sum_points,p),dtype='complex')
    A_train = np.zeros((num_train_subjects*sum_points,p))
    A_test = np.zeros((num_test_subjects*sum_points,p))
    U_cos_train = np.zeros((num_train_subjects*sum_points,p,2))
    U_cos_test = np.zeros((num_test_subjects*sum_points,p,2))
    L_cos_train = np.zeros((num_train_subjects*sum_points,2))
    L_cos_test = np.zeros((num_test_subjects*sum_points,2))
    timeseries_train = np.zeros((num_train_subjects*sum_points,p))
    timeseries_test = np.zeros((num_test_subjects*sum_points,p))
    for j,task in enumerate(tasks):
        if split_6040_or_8020 == '6040':
            tmp_filename = 'paper/data/processed/concatenated_datasets/'+add_preproc+task+'fMRI_'+atlas+add_GSR+'.h5'
        elif split_6040_or_8020 == '8020':
            tmp_filename = 'paper/data/processed/concatenated_datasets/'+add_preproc+task+'fMRI_'+atlas+add_GSR+'_8020split'+str(split_idx)+'.h5'
        train_count = 0
        test_count = 0
        with h5.File(tmp_filename,'r') as f_task:
            for i,sub in enumerate(subjects):
                print(task,i)
                if sub in train_subjects:  # subject_split
                    U_complex_train[train_count*sum_points+cumsum_points[j]:train_count*sum_points+cumsum_points[j+1]] = f_task['U_complex_train'][:][train_count*num_points_after_removal[j]:(train_count+1)*num_points_after_removal[j]]
                    A_train[train_count*sum_points+cumsum_points[j]:train_count*sum_points+cumsum_points[j+1]] = f_task['A_train'][:][train_count*num_points_after_removal[j]:(train_count+1)*num_points_after_removal[j]]
                    U_cos_train[train_count*sum_points+cumsum_points[j]:train_count*sum_points+cumsum_points[j+1]] = f_task['U_cos_train'][:][train_count*num_points_after_removal[j]:(train_count+1)*num_points_after_removal[j]]
                    L_cos_train[train_count*sum_points+cumsum_points[j]:train_count*sum_points+cumsum_points[j+1]] = f_task['L_cos_train'][:][train_count*num_points_after_removal[j]:(train_count+1)*num_points_after_removal[j]]
                    timeseries_train[train_count*sum_points+cumsum_points[j]:train_count*sum_points+cumsum_points[j+1]] = f_task['timeseries_train'][:][train_count*num_points_after_removal[j]:(train_count+1)*num_points_after_removal[j]]
                    train_count += 1
                elif sub in test_subjects: # subject_split
                    U_complex_test[test_count*sum_points+cumsum_points[j]:test_count*sum_points+cumsum_points[j+1]] = f_task['U_complex_test'][:][test_count*num_points_after_removal[j]:(test_count+1)*num_points_after_removal[j]]
                    A_test[test_count*sum_points+cumsum_points[j]:test_count*sum_points+cumsum_points[j+1]] = f_task['A_test'][:][test_count*num_points_after_removal[j]:(test_count+1)*num_points_after_removal[j]]
                    U_cos_test[test_count*sum_points+cumsum_points[j]:test_count*sum_points+cumsum_points[j+1]] = f_task['U_cos_test'][:][test_count*num_points_after_removal[j]:(test_count+1)*num_points_after_removal[j]]
                    L_cos_test[test_count*sum_points+cumsum_points[j]:test_count*sum_points+cumsum_points[j+1]] = f_task['L_cos_test'][:][test_count*num_points_after_removal[j]:(test_count+1)*num_points_after_removal[j]]
                    timeseries_test[test_count*sum_points+cumsum_points[j]:test_count*sum_points+cumsum_points[j+1]] = f_task['timeseries_test'][:][test_count*num_points_after_removal[j]:(test_count+1)*num_points_after_removal[j]]
                    test_count += 1
                else:
                    raise ValueError(f"Subject {sub} not found in either train or test subjects.")
        
    f.create_dataset('U_complex_train',data=U_complex_train)
    f.create_dataset('U_complex_test',data=U_complex_test)
    f.create_dataset('A_train',data=A_train)
    f.create_dataset('A_test',data=A_test)
    f.create_dataset('U_cos_train',data=U_cos_train)
    f.create_dataset('U_cos_test',data=U_cos_test)
    f.create_dataset('L_cos_train',data=L_cos_train)
    f.create_dataset('L_cos_test',data=L_cos_test)
    f.create_dataset('timeseries_train',data=timeseries_train)
    f.create_dataset('timeseries_test',data=timeseries_test)
    

# # ################### REST
# add_GSR = '_GSR'
# # add_GSR = ''
# dataset = 'Atlas_MSMAll_hp2000_clean'
# atlas = 'SchaeferTian116'
# task1 = 'REST1'
# task2 = 'REST2'
# num_points = np.array([1200,1200])
# subject_split = 155
# sum_num_points = np.sum(num_points)
# num_subs = len(subjects)
# complex_eigenvectors_train = np.zeros((sum_num_points//2*subject_split,p,1),dtype='complex')
# complex_eigenvectors_test1 = np.zeros((sum_num_points//2*subject_split,p,1),dtype='complex')
# complex_eigenvectors_test2 = np.zeros((sum_num_points*(num_subs-subject_split),p,1),dtype='complex')
# hilbert_amplitude_train = np.zeros((sum_num_points//2*subject_split,p))
# hilbert_amplitude_test1 = np.zeros((sum_num_points//2*subject_split,p))
# hilbert_amplitude_test2 = np.zeros((sum_num_points*(num_subs-subject_split),p))
# cos_eigenvectors_train = np.zeros((sum_num_points//2*subject_split,p,2))
# cos_eigenvectors_test1 = np.zeros((sum_num_points//2*subject_split,p,2))
# cos_eigenvectors_test2 = np.zeros((sum_num_points*(num_subs-subject_split),p,2))
# cos_eigenvalues_train = np.zeros((sum_num_points//2*subject_split,2))
# cos_eigenvalues_test1 = np.zeros((sum_num_points//2*subject_split,2))
# cos_eigenvalues_test2 = np.zeros((sum_num_points*(num_subs-subject_split),2))
# timeseries_train = np.zeros((sum_num_points//2*subject_split,p,1))
# timeseries_test1 = np.zeros((sum_num_points//2*subject_split,p,1))
# timeseries_test2 = np.zeros((sum_num_points*(num_subs-subject_split),p,1))
# #shuffle subjects
# np.random.seed(0)
# np.random.shuffle(subjects)
# # save the subject split
# np.savetxt('paper/data/REST1REST2_subject_split.txt',subjects,fmt='%s')
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
#         evecs_real = np.loadtxt('paper/data/processed/RESTfMRI_'+atlas+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_'+dataset+'_real.csv',delimiter=',')
#         evecs_imag = np.loadtxt('paper/data/processed/RESTfMRI_'+atlas+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_'+dataset+'_imag.csv',delimiter=',')
#         evecs_complex = evecs_real + 1j*evecs_imag
#         hilbert_amplitude = np.loadtxt('paper/data/processed/RESTfMRI_'+atlas+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_'+dataset+'_amplitude.csv',delimiter=',')
#         tmp_evecs = np.loadtxt('paper/data/processed/RESTfMRI_'+atlas+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_'+dataset+'.csv',delimiter=',')
#         evecs_cos = np.zeros((num_points[j],p,2))
#         evecs_cos[:,:,0] = tmp_evecs[::2,:]
#         evecs_cos[:,:,1] = tmp_evecs[1::2,:]
#         tmp_evals = np.loadtxt('paper/data/processed/RESTfMRI_'+atlas+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_'+dataset+'_evs.csv',delimiter=',')
#         evals_cos = np.zeros((num_points[j],2))
#         evals_cos[:,0] = tmp_evals[::2]
#         evals_cos[:,1] = tmp_evals[1::2]
#         timeseries = np.loadtxt('paper/data/processed/RESTfMRI_'+atlas+add_GSR+'/'+str(sub)+'_rfMRI_'+task+'_RL_'+dataset+'_timeseries.csv',delimiter=',')
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
    
# with h5.File('paper/data/processed/'+task1+task2+'fMRI_'+atlas+add_GSR+'.h5','w') as f:
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