import pandas as pd
import numpy as np
import h5py as h5
from PCMM.helper_functions import train_model,test_model
from PCMM.supervised_models import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f
import warnings
from paper.extract_first_N_poststimulus_volumes import extract_first_N_poststim_volumes

def load_fMRI_data(data_file,options, standardize=False, covariance=False):
    assert options['modelname'] in ['Watson','ACG','MACG','SingularWishart','Complex_Watson',
                                    'Complex_ACG','Normal','Complex_Normal',
                                    'least_squares','diametrical','complex_diametrical','grassmann','weighted_grassmann',
                                    'linear-svm','rbf-svm','logistic']

    if options['experiment'] == 'all_tasks':
        with h5.File(data_file,'r') as f:
            if options['modelname'] in ['Complex_Watson','Complex_ACG','complex_diametrical']:
                # complex normalized phase vectors
                data_train = f['U_complex_train'][:]
                data_test = f['U_complex_test'][:]
            elif options['modelname'] in ['Watson','ACG','least_squares','diametrical']:
                # leading eigenvector of cosinus phase coherence matrix
                data_train = f['U_cos_train'][:][:,:,0]
                data_test = f['U_cos_test'][:][:,:,0]
            elif options['modelname'] in ['MACG','grassmann']:
                # both eigenvectors of cosinus phase coherence matrix
                data_train = f['U_cos_train'][:]
                data_test = f['U_cos_test'][:]
            elif options['modelname'] in ['SingularWishart','weighted_grassmann']:
                # both eigenvectors and both eigenvalues of cosinus phase coherence matrix
                data_train = f['U_cos_train'][:]*np.sqrt(f['L_cos_train'][:][:,None,:])
                data_test = f['U_cos_test'][:]*np.sqrt(f['L_cos_test'][:][:,None,:])
            elif options['modelname'] in ['Complex_Normal']:
                # complex normalized phase vectors scaled by hilbert amplitude
                data_train = f['U_complex_train'][:]*f['A_train'][:]
                data_test = f['U_complex_test'][:]*f['A_test'][:]
            elif options['modelname'] in ['Normal','linear-svm','rbf-svm','logistic']:
                # filtered time series data (no Hilbert transform)
                data_train = f['timeseries_train'][:]
                data_test = f['timeseries_test'][:]
            else:
                raise ValueError("Problem")

        # standardize data for each subject
        if options['experiment'] == 'all_tasks':
            num_pts_per_subject = [176,253,316,284,232,274,405]
            sum_num_pts_per_subject = np.sum(num_pts_per_subject)
        else:
            raise ValueError('Problem')
        if standardize:
            
            for sub in range(data_train.shape[0]//sum_num_pts_per_subject):
                for task in range(7):
                    data_train[sub*sum_num_pts_per_subject+sum(num_pts_per_subject[:task]):sub*sum_num_pts_per_subject+sum(num_pts_per_subject[:task+1])] = StandardScaler().fit_transform(data_train[sub*sum_num_pts_per_subject+sum(num_pts_per_subject[:task]):sub*sum_num_pts_per_subject+sum(num_pts_per_subject[:task+1])])
            for sub in range(data_test.shape[0]//sum_num_pts_per_subject):
                for task in range(7):
                    data_test[sub*sum_num_pts_per_subject+sum(num_pts_per_subject[:task]):sub*sum_num_pts_per_subject+sum(num_pts_per_subject[:task+1])] = StandardScaler().fit_transform(data_test[sub*sum_num_pts_per_subject+sum(num_pts_per_subject[:task]):sub*sum_num_pts_per_subject+sum(num_pts_per_subject[:task+1])])
        
        if options.get('first_N_poststim_volumes') not in ['cov','all',None]:
            num_pts_per_task = np.array([176,253,316,284,232,274,405])
            cumsum_pts_per_task = np.concatenate([np.zeros(1),np.cumsum(num_pts_per_task)]).astype(int)
            num_pts_per_subject = np.sum(num_pts_per_task)
            data_train2 = []
            data_test2 = []
            subjectlist = np.loadtxt('paper/data/all_tasks_subject_split.txt',dtype=str)
            subjects_train = subjectlist[:155]
            subjects_test = subjectlist[155:]
            tasks = ['EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM']
            print('\n')
            true_labels_train = []
            true_labels_test = []
            pts_per_subject_train = []
            pts_per_subject_test = []
            for sub in range(155):
                pts_per_task = []
                for task in range(7):
                    print('Extracting first',options['first_N_poststim_volumes'],'post-stimulus volumes for train subject',sub+1,end='\r')
                    tmp = data_train[sub*num_pts_per_subject+cumsum_pts_per_task[task]:sub*num_pts_per_subject+cumsum_pts_per_task[task+1]]
                    tmp2 = extract_first_N_poststim_volumes(data=tmp,subject=subjects_train[sub],task=tasks[task],N=options['first_N_poststim_volumes'],tr=0.72,first_poststimulus_volume=1)[0]
                    data_train2.append(tmp2)
                    pts_per_task.append(np.shape(tmp2)[0])
                    true_labels_train.append(np.shape(tmp2)[0]*[task])
                pts_per_subject_train.append(np.array(pts_per_task))
            print('\n')
            for sub in range(99):
                pts_per_task = []
                for task in range(7):
                    print('Extracting first',options['first_N_poststim_volumes'],'post-stimulus volumes for test subject',sub+1,end='\r')
                    tmp = data_test[sub*num_pts_per_subject+cumsum_pts_per_task[task]:sub*num_pts_per_subject+cumsum_pts_per_task[task+1]]
                    tmp2 = extract_first_N_poststim_volumes(data=tmp,subject=subjects_test[sub],task=tasks[task],N=options['first_N_poststim_volumes'],tr=0.72,first_poststimulus_volume=1)[0]
                    data_test2.append(tmp2)
                    true_labels_test.append(np.shape(tmp2)[0]*[task])
                    pts_per_task.append(np.shape(tmp2)[0])
                pts_per_subject_test.append(np.array(pts_per_task))
            data_train = np.concatenate(data_train2,axis=0)
            data_test = np.concatenate(data_test2,axis=0)
            true_labels_train = np.concatenate(true_labels_train,axis=0)
            true_labels_test = np.concatenate(true_labels_test,axis=0)
            return data_train,data_test[0:2],data_test, true_labels_train, true_labels_test, pts_per_subject_train, pts_per_subject_test

        if covariance:
            if options.get('first_N_poststim_volumes') is not None:
                raise ValueError('Covariance not implemented with first N poststim volumes option')
            data_train_cov = []
            data_test_cov = []
            for sub in range(data_train.shape[0]//sum_num_pts_per_subject):
                for task in range(7):
                    data_sub_task = data_train[sub*sum_num_pts_per_subject+sum(num_pts_per_subject[:task]):sub*sum_num_pts_per_subject+sum(num_pts_per_subject[:task+1])]
                    cov_mat = np.cov(data_sub_task,rowvar=False)
                    cov_mat = cov_mat[np.triu_indices(cov_mat.shape[0],k=1)]
                    data_train_cov.append(cov_mat.flatten())
            for sub in range(data_test.shape[0]//sum_num_pts_per_subject):
                for task in range(7):
                    data_sub_task = data_test[sub*sum_num_pts_per_subject+sum(num_pts_per_subject[:task]):sub*sum_num_pts_per_subject+sum(num_pts_per_subject[:task+1])]
                    cov_mat = np.cov(data_sub_task,rowvar=False)
                    cov_mat = cov_mat[np.triu_indices(cov_mat.shape[0],k=1)]
                    data_test_cov.append(cov_mat.flatten())
            data_train = np.array(data_train_cov)
            data_test = np.array(data_test_cov)
        
        return data_train,data_test[0:2],data_test
    # else if options['experiment'] == 'REST1REST2':
    with h5.File(data_file,'r') as f:
        if options['modelname'] in ['Complex_Watson','Complex_ACG','complex_diametrical']:
            # complex normalized phase vectors
            data_train = f['U_complex_train'][:][:,:,0]
            data_test1 = f['U_complex_test1'][:][:,:,0]
            data_test2 = f['U_complex_test2'][:][:,:,0] 
        elif options['modelname'] in ['Watson','ACG','least_squares','diametrical']:
            # leading eigenvector of cosinus phase coherence matrix
            data_train = f['U_cos_train'][:][:,:,0]
            data_test1 = f['U_cos_test1'][:][:,:,0]
            data_test2 = f['U_cos_test2'][:][:,:,0]
        elif options['modelname'] in ['MACG','grassmann']:
            # both eigenvectors of cosinus phase coherence matrix
            data_train = f['U_cos_train'][:]
            data_test1 = f['U_cos_test1'][:]
            data_test2 = f['U_cos_test2'][:]
        elif options['modelname'] in ['SingularWishart','weighted_grassmann']:
            # both eigenvectors and both eigenvalues of cosinus phase coherence matrix
            data_train = f['U_cos_train'][:]*np.sqrt(f['L_cos_train'][:][:,None,:])
            data_test1 = f['U_cos_test1'][:]*np.sqrt(f['L_cos_test1'][:][:,None,:])
            data_test2 = f['U_cos_test2'][:]*np.sqrt(f['L_cos_test2'][:][:,None,:])
        elif options['modelname'] in ['Complex_Normal']:
            # complex normalized phase vectors scaled by hilbert amplitude
            data_train = f['U_complex_train'][:][:,:,0]*f['A_train'][:]
            data_test1 = f['U_complex_test1'][:][:,:,0]*f['A_test1'][:]
            data_test2 = f['U_complex_test2'][:][:,:,0]*f['A_test2'][:]
        elif options['modelname'] in ['Normal']:
            # filtered time series data (no Hilbert transform)
            data_train = f['timeseries_train'][:][:,:,0]
            data_test1 = f['timeseries_test1'][:][:,:,0]
            data_test2 = f['timeseries_test2'][:][:,:,0]
        else:
            raise ValueError("Problem")
    return data_train,data_test1,data_test2

# from sklearn.mixture import GaussianMixture
# gm = GaussianMixture(n_components=7, covariance_type='full', max_iter=100000, tol=1e-6, verbose=2).fit(data_train)
# train_posterior = gm.predict_proba(data_train).T
# test1_posterior = gm.predict_proba(data_test1).T
# test2_posterior = gm.predict_proba(data_test2).T

def run(data_train,data_test1,data_test2,K,df,options,params=None,suppress_output=False,inner=None,p=116,true_labels_train=None,true_labels_test=None, pts_per_subject_train=None, pts_per_subject_test=None,train_or_not=True):

    if data_train.shape[1]==6670: # covariance features as input
        samples_per_sequence = [1,1,1]
        # repeat 0..6 for each subject
        true_labels_int = [np.hstack(155*[np.arange(7)]),np.hstack(155*[np.arange(7)]),np.hstack(99*[np.arange(7)])]
        num_subs = [155,155,99]
        pts_pr_subject_sum = np.array([7,7,7])
        num_pts_per_subject = [pts_pr_subject_sum[0],pts_pr_subject_sum[1],pts_pr_subject_sum[2]]
        reduced_number_of_points = True
        cumsum_pts_pr_task = np.array([np.arange(8), np.arange(8), np.arange(8)])
        true_labels = [np.zeros((7,pts_pr_subject_sum[0])),np.zeros((7,pts_pr_subject_sum[1])),np.zeros((7,pts_pr_subject_sum[2]))]
        for task in range(7):
            true_labels[0][task,task::7] = 1
            true_labels[1][task,task::7] = 1
            true_labels[2][task,task::7] = 1

    elif options.get('first_N_poststim_volumes') not in ['cov','all',None]: # first N post-stim volumes
        reduced_number_of_points = True
        samples_per_sequence = [options['first_N_poststim_volumes'],options['first_N_poststim_volumes'],options['first_N_poststim_volumes']]
        true_labels_int = [true_labels_train, true_labels_train, true_labels_test]
        num_subs = [155,155,99]
        num_pts_per_task = [pts_per_subject_train, pts_per_subject_train, pts_per_subject_test]
        num_pts_per_subject = [np.sum(np.array(pts_per_subject_train),axis=1),np.sum(np.array(pts_per_subject_train),axis=1),np.sum(np.array(pts_per_subject_test),axis=1)]
        cumsum_pts_pr_task = [np.zeros((num_subs[0],8)).astype(int),np.zeros((num_subs[1],8)).astype(int),np.zeros((num_subs[2],8)).astype(int)]
        cumsum_pts_pr_task = [np.concatenate([np.zeros(1),np.cumsum(num_pts_per_subject[0])]).astype(int),
                                    np.concatenate([np.zeros(1),np.cumsum(num_pts_per_subject[1])]).astype(int),
                                    np.concatenate([np.zeros(1),np.cumsum(num_pts_per_subject[2])]).astype(int)]
        true_labels = [np.zeros((7,data_train.shape[0])),np.zeros((7,data_train.shape[0])),np.zeros((7,data_test2.shape[0]))]
        pts_pr_subject_sum = [np.zeros(num_subs[0]),np.zeros(num_subs[1]),np.zeros(num_subs[2])]
        for Set in range(3):
            for sub in range(num_subs[Set]):
                cumsum_pts_pr_task[Set][sub,1:] = np.cumsum(num_pts_per_task[Set][sub]).astype(int)
                pts_pr_subject_sum[Set][sub] = np.sum(num_pts_per_task[Set][sub])
                true_labels_sub = true_labels_int[Set][cumsum_pts_pr_task[Set][sub]:cumsum_pts_pr_task[Set][sub+1]]
                for task in range(7):
                    true_labels[Set][task,cumsum_pts_pr_task[Set][sub]:cumsum_pts_pr_task[Set][sub+1]][true_labels_sub==task] = 1
    
    elif options['experiment'] == 'all_tasks':
        if 'split' in options['outfolder']:
            num_pts_per_subject = data_train.shape[0]//203
        else:
            num_pts_per_subject = data_train.shape[0]//155
        tasks = ['EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM']
        reduced_number_of_points = False

        if num_pts_per_subject == 1870: # minus [15,11,0,11,11,11,11]
            samples_per_sequence = [[176-15,253-11,316,284-11,232-11,274-11,405-11],0,[176-15,253-11,316,284-11,232-11,274-11,405-11]]
            pts_pr_subject = [[0,176-15,253-11,316,284-11,232-11,274-11,405-11],[0,0,0,0,0,0,0,0],[0,176-15,253-11,316,284-11,232-11,274-11,405-11]]
            num_subs = [data_train.shape[0]//1870,0,data_test2.shape[0]//1870]
            pts_pr_subject_sum = np.array([1870,0,1870])

            cumsum_pts_pr_task = np.cumsum(pts_pr_subject,axis=1)
            true_labels = [np.zeros((7,pts_pr_subject_sum[0])),np.zeros((7,pts_pr_subject_sum[1])),np.zeros((7,pts_pr_subject_sum[2]))]
            for task in range(7):
                task_indices = np.loadtxt('paper/data/task_indices/100206_'+tasks[task]+'_task_vector.txt')
                if tasks[task] in ['EMOTION']:
                    task_indices = task_indices[15:]
                elif tasks[task] in ['GAMBLING','MOTOR','RELATIONAL','SOCIAL','WM']:
                    task_indices = task_indices[11:]
                true_labels[0][task,cumsum_pts_pr_task[0][task]+np.where(task_indices)] = 1
                # true_labels[1][task,cumsum_pts_pr_subject[1][task]+np.where(task_indices)] = 1
                true_labels[2][task,cumsum_pts_pr_task[2][task]+np.where(task_indices)] = 1
            # true_labels_int = [np.hstack(155*[np.argmax(true_labels[0],axis=0)+1]),np.hstack(155*[np.argmax(true_labels[1],axis=0)]),np.hstack(155*[np.argmax(true_labels[2],axis=0)+1])]
            result = np.argmax(true_labels[0], axis=0) + 1
            result[np.sum(true_labels[0], axis=0) == 0] = 0
            true_labels_int = [result,np.hstack(155*[np.argmax(true_labels[1],axis=0)]),result]
        else:
            samples_per_sequence = [[176,253,316,284,232,274,405],0,[176,253,316,284,232,274,405]]
            pts_pr_subject = [[0,176,253,316,284,232,274,405],[0,0,0,0,0,0,0,0],[0,176,253,316,284,232,274,405]]
            num_subs = [data_train.shape[0]//1940,0,data_test2.shape[0]//1940]
            pts_pr_subject_sum = np.array([1940,0,1940])

            cumsum_pts_pr_task = np.cumsum(pts_pr_subject,axis=1)
            true_labels = [np.zeros((7,pts_pr_subject_sum[0])),np.zeros((7,pts_pr_subject_sum[1])),np.zeros((7,pts_pr_subject_sum[2]))]
            for task in range(7):
                true_labels[0][task,cumsum_pts_pr_task[0][task]:cumsum_pts_pr_task[0][task+1]] = 1
                true_labels[1][task,cumsum_pts_pr_task[1][task]:cumsum_pts_pr_task[1][task+1]] = 1
                true_labels[2][task,cumsum_pts_pr_task[2][task]:cumsum_pts_pr_task[2][task+1]] = 1
            true_labels_int = [np.hstack(155*[np.argmax(true_labels[0],axis=0)]),np.hstack(155*[np.argmax(true_labels[1],axis=0)]),np.hstack(155*[np.argmax(true_labels[2],axis=0)])]

    else: # resting-state
        pts_pr_subject_sum = np.array([1200,1200,2400])
        samples_per_sequence = [1200,1200,2400]
        num_subs = [data_train.shape[0]//1200,data_test1.shape[0]//1200,data_test2.shape[0]//2400]
    
    # do training if supervised models
    if train_or_not:
        if options['modelname'] == 'linear-svm':
            params,train_posterior,test2_posterior = svm_linear_cv(train_X=data_train,train_y=true_labels_int[0],test_X=data_test2,
                                samples_per_subject_train=None)
            test1_posterior = np.zeros_like(train_posterior)
            loglik_curve = []
        elif options['modelname'] == 'rbf-svm':
            params,train_posterior,test2_posterior = svm_rbf_cv(train_X=data_train,train_y=true_labels_int[0],test_X=data_test2,
                            samples_per_subject_train=None)
            test1_posterior = np.zeros_like(train_posterior)
            loglik_curve = []
        elif options['modelname'] == 'logistic':
            params,train_posterior,test2_posterior = logistic_l2_cv(train_X=data_train,train_y=true_labels_int[0],test_X=data_test2,
                                samples_per_subject_train=None) #155*[np.sum(samples_per_sequence[0])])
            test1_posterior = np.zeros_like(train_posterior)
            loglik_curve = []
        else: #mixtures
            params,train_posterior,loglik_curve = train_model(data_train,K=K,options=options,suppress_output=suppress_output,samples_per_sequence=samples_per_sequence[0],params=params)
            
            train_loglik,train_posterior,train_loglik_per_sample = test_model(data_test=data_train,params=params,K=K,options=options,samples_per_sequence=samples_per_sequence[0])
            test1_loglik,test1_posterior,test1_loglik_per_sample = test_model(data_test=data_test1,params=params,K=K,options=options,samples_per_sequence=samples_per_sequence[1])
            test2_loglik,test2_posterior,test2_loglik_per_sample = test_model(data_test=data_test2,params=params,K=K,options=options,samples_per_sequence=samples_per_sequence[2])
        
    else: # skip training and use provided params
        if options['modelname'] in ['linear-svm','rbf-svm','logistic']:
            raise ValueError('Cannot skip training for supervised models')
        train_loglik,train_posterior,train_loglik_per_sample = test_model(data_test=data_train,params=params,K=K,options=options,samples_per_sequence=samples_per_sequence[0])
        test1_loglik,test1_posterior,test1_loglik_per_sample = test_model(data_test=data_test1,params=params,K=K,options=options,samples_per_sequence=samples_per_sequence[1])
        test2_loglik,test2_posterior,test2_loglik_per_sample = test_model(data_test=data_test2,params=params,K=K,options=options,samples_per_sequence=samples_per_sequence[2])
        loglik_curve = []
    posteriors = [train_posterior,test1_posterior,test2_posterior]

    if options['HMM']:
        train_ll = train_loglik_per_sample/np.sum(samples_per_sequence[0])
        test1_ll = test1_loglik_per_sample/np.sum(samples_per_sequence[1])
        test2_ll = test2_loglik_per_sample/np.sum(samples_per_sequence[2])
    else:
        if options['modelname'] in ['linear-svm','rbf-svm','logistic']:
            train_ll = np.zeros(num_subs[0])
            test1_ll = np.zeros(num_subs[1])
            test2_ll = np.zeros(num_subs[2])
        else:
            train_ll,test1_ll,test2_ll = calc_ll_per_sub(train_loglik_per_sample,test1_loglik_per_sample,test2_loglik_per_sample,options)
    end_logliks = [train_ll,test1_ll,test2_ll]
    sets = ['train','test1','test2']

    comp_order = None
    if not train_or_not:
        if options['modelname']=='Complex_ACG':
            comp_order = [0,3,4,2,1,6,5]
        elif options['modelname']=='Complex_Normal':
            comp_order = [2,6,1,4,3,5,0]
        elif options['modelname']=='Normal':
            comp_order = [0,2,5,1,6,3,4]
    elif options['modelname'] not in ['linear-svm','rbf-svm','logistic'] and options['first_N_poststim_volumes'] is not None: #mixtures
        # average the train posterior first within subject, then across subjects to get a component order
        sub_posteriors = []
        for i in range(num_subs[0]):
            posterior_sub = train_posterior[:,num_pts_per_subject*i:num_pts_per_subject*(i+1)]
            # average within subject based on task lengths
            posterior_task = np.zeros((K,7))
            for task in range(7):
                posterior_task[:,task] = np.nanmean(posterior_sub[:,cumsum_pts_pr_task[0][task]:cumsum_pts_pr_task[0][task+1]],axis=1)
            sub_posteriors.append(posterior_task)
        mean_posteriors = np.mean(np.stack(sub_posteriors,axis=2),axis=2) #Kx7
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-mean_posteriors)
        comp_order = row_ind[np.argsort(col_ind)]
        # reorder posteriors based on this component order and params
        train_posterior = train_posterior[comp_order,:]
        # test1_posterior = test1_posterior[comp_order,:] 
        test2_posterior = test2_posterior[comp_order,:]
        posteriors = [train_posterior,test1_posterior,test2_posterior]
        param_keys = list(params.keys())
        for key in param_keys:
            params[key] = params[key][comp_order]

    entries = []
    all_test_posterior = []
    all_train_posterior = []
    for set in range(len(sets)):
        for i in range(num_subs[set]):
            if options['experiment'] == 'all_tasks':
                if set==1:
                    continue
                if reduced_number_of_points:
                    posterior_sub = np.zeros((K,num_pts_per_subject[set][i]))
                    for task in range(7):
                        posterior_sub[:,cumsum_pts_pr_task[set][i][task]:cumsum_pts_pr_task[set][i][task+1]] = posteriors[set][:,cumsum_pts_pr_task[set][i]+cumsum_pts_pr_task[set][i][task]:cumsum_pts_pr_task[set][i]+cumsum_pts_pr_task[set][i][task+1]]
                    nmi = calc_NMI(posterior_sub,true_labels[set][:,cumsum_pts_pr_task[set][i]:cumsum_pts_pr_task[set][i+1]])
                    # calculate classification accuracy, assume that component one corresponds to task one etc
                    # if comp_order is not None:
                    #     posterior_sub = posterior_sub[comp_order,:]
                    predicted_labels = np.argmax(posterior_sub,axis=0)
                    true_labels_sub = true_labels_int[set][cumsum_pts_pr_task[set][i]:cumsum_pts_pr_task[set][i+1]]
                    classification_accuracy = np.mean(predicted_labels==true_labels_sub)
                else:
                    posterior_sub = np.zeros((K,pts_pr_subject_sum[set]))
                    for task in range(7):
                        posterior_sub[:,cumsum_pts_pr_task[set][task]:cumsum_pts_pr_task[set][task+1]] = posteriors[set][:,i*pts_pr_subject_sum[set]+cumsum_pts_pr_task[set][task]:i*pts_pr_subject_sum[set]+cumsum_pts_pr_task[set][task+1]]
                    # posterior = posteriors[set][:,i*pts_pr_subject_sum[set]:(i+1)*pts_pr_subject_sum[set]]
                    if len(np.unique(true_labels_int[set]))==8:
                        nonzero_idx = true_labels_int[set]>0
                        nmi = calc_NMI(posterior_sub[:,nonzero_idx],true_labels[set][:,nonzero_idx])
                    else:
                        nmi = calc_NMI(posterior_sub,true_labels[set])
                    # if comp_order is not None:
                    #     posterior_sub = posterior_sub[comp_order,:]
                    if options['modelname'] in ['linear-svm','rbf-svm','logistic'] or train_or_not==False:
                        predicted_labels = np.argmax(posterior_sub,axis=0)
                        true_labels_sub = true_labels_int[set][i*pts_pr_subject_sum[set]:(i+1)*pts_pr_subject_sum[set]]
                        classification_accuracy = np.mean(predicted_labels==true_labels_sub)
                    else:
                        classification_accuracy = np.nan
            else:
                nmi = np.nan
                classification_accuracy = np.nan
                posterior_sub = posteriors[set][:,i*pts_pr_subject_sum[set]:(i+1)*pts_pr_subject_sum[set]]
            entropy = -np.nansum(posterior_sub * np.log(posterior_sub + 1e-10), axis=(-1,-2)) #entropy per time point
            pi_sub = np.nanmean(posterior_sub, axis=1) #mean over time points
            entropy_pi = -np.nansum(pi_sub * np.log(pi_sub + 1e-10)) #entropy of mean posterior
            entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),
                'K':K,'p':p,'rank':options['rank'],'inner':inner,'iter':len(loglik_curve),
                'Set':sets[set],'loglik':end_logliks[set][i],'Subject':i,'NMI':nmi,'classification_accuracy':classification_accuracy,
                'entropy':entropy,'entropy_pi':entropy_pi}
            entries.append(entry)
            if set==0:
                all_train_posterior.append(posterior_sub)
            if set==2:
                all_test_posterior.append(posterior_sub)


    # accs, accs_ensemble = posterior_to_accuracy(
    #     train_posterior,
    #     test2_posterior,
    #     cov_or_ts='ts',
    # )

    df = pd.concat([df,pd.DataFrame(entries)],ignore_index=True)
    print(options['first_N_poststim_volumes'],df[df['Set']=='test2']['classification_accuracy'].mean())
    print(options['first_N_poststim_volumes'],df[df['Set']=='test2']['NMI'].mean())
    return params,df,np.concatenate(all_train_posterior,axis=1),np.concatenate(all_test_posterior,axis=1)

def calc_ll_per_sub(train_loglik_per_sample,test1_loglik_per_sample,test2_loglik_per_sample,options):
    if train_loglik_per_sample.shape[0]==155:
        return train_loglik_per_sample,test1_loglik_per_sample,test2_loglik_per_sample
    if options['experiment'] == 'all_tasks':
        if 'split' in options['outfolder']:
            num_subs = [203,0,51]
        else:
            num_subs = [155,0,99]
        # num_subs = [155,0,99]
        num_samples_per_sub_train = train_loglik_per_sample.shape[0]//num_subs[0]
        num_samples_per_sub_test1 = test1_loglik_per_sample.shape[0]
        num_samples_per_sub_test2 = test2_loglik_per_sample.shape[0]//num_subs[2]
    else:
        num_subs = [155,155,100]
        num_samples_per_sub_train = 1200
        num_samples_per_sub_test1 = 1200
        num_samples_per_sub_test2 = 2400

    train_loglik_per_sub = np.zeros(num_subs[0])
    test1_loglik_per_sub = np.zeros(num_subs[1])
    test2_loglik_per_sub = np.zeros(num_subs[2])

    for i in range(np.max(num_subs)):
        train_loglik_per_sub[i] = np.mean(train_loglik_per_sample[i*num_samples_per_sub_train:(i+1)*num_samples_per_sub_train])
        if i<num_subs[1]:
            test1_loglik_per_sub[i] = np.mean(test1_loglik_per_sample[i*num_samples_per_sub_test1:(i+1)*num_samples_per_sub_test1])
        if i<num_subs[2]:
            test2_loglik_per_sub[i] = np.mean(test2_loglik_per_sample[i*num_samples_per_sub_test2:(i+1)*num_samples_per_sub_test2])

    return train_loglik_per_sub,test1_loglik_per_sub,test2_loglik_per_sub
        
def calc_MI(Z1,Z2):
    P=Z1@Z2.T
    PXY=P/np.sum(P)
    if np.isclose(np.sum(P),0):
        a=7
    PXPY=np.outer(np.sum(PXY,axis=1),np.sum(PXY,axis=0))
    ind=np.where(PXY>0) #PXY should always be >0
    MI=np.sum(PXY[ind]*np.log(PXY[ind]/PXPY[ind]))
    return MI

def calc_NMI(Z1,Z2):
    #Z1 and Z2 are two partition matrices of size (K1xN) and (K2xN) where K is number of components and N is number of samples
    NMI = (2*calc_MI(Z1,Z2))/(calc_MI(Z1,Z1)+calc_MI(Z2,Z2))
    return NMI

def make_true_mat(num_subs=10,K=5):
    rows = []
    for _ in range(num_subs):
        row = np.zeros((K,1200),dtype=bool)
        num_samples_per_cluster = 1200//K
        for k in range(K):
            row[k,num_samples_per_cluster*k:num_samples_per_cluster*(k+1)] = True
        rows.append(row)
    return np.hstack(rows)

def horizontal_boxplot(df_fig,type=1,ranks=[1,10,25]):
    order = ['Mixture: Complex Watson',
        *['Mixture: Complex ACG rank='+str(rank) for rank in ranks],
        'K-means: Complex diametrical','space1','space2',
        *['Mixture: MACG rank='+str(rank) for rank in ranks],
        *['Mixture: Singular Wishart rank='+str(rank) for rank in ranks],
        'K-means: Grassmann','K-means: Weighted Grassmann','space3','space4',
        'Mixture: Watson',
        *['Mixture: ACG rank='+str(rank) for rank in ranks],
        'K-means: Diametrical','K-means: Least squares (sign-flip)','space5','space6',
        *['Mixture: Gaussian rank='+str(rank) for rank in ranks],'space7','space8',
        *['Mixture: Complex Gaussian rank='+str(rank) for rank in ranks]]
    palette_husl = sns.color_palette("husl", n_colors=11, desat=1)
    palette_husl.append((0.5,0.5,0.5))
    palette_husl.append((0.3,0.3,0.3))
    palette_husl2 = [palette_husl[0]]+[palette_husl[1]]*len(ranks)+[palette_husl[2]]+[palette_husl[-1]]*2+[palette_husl[3]]*len(ranks)+[palette_husl[4]]*len(ranks)+[palette_husl[5]]+[palette_husl[6]]+[palette_husl[-1]]*2+[palette_husl[7]]+[palette_husl[8]]*len(ranks)+[palette_husl[9]]+[palette_husl[10]]+[palette_husl[-1]]*2+[palette_husl[11]]*len(ranks)+[palette_husl[-1]]*2+[palette_husl[12]]*len(ranks)
    # df_fig = df[df['Set']=='Out-of-sample test']
    for i in range(1,9):
        df_fig2 = pd.concat([df_fig,pd.DataFrame({'NMI':[np.nan],'names2':['space'+str(i)]}, index=[0])], ignore_index=True)
    fig = plt.figure(figsize=(10,7))
    if type == 1:
        sns.boxplot(x='NMI', y='names2', data=df_fig2, palette=palette_husl2, order=order)
        plt.xlabel('Normalized mutual information')
        plt.xlim([-0.01,1.01])
        xtitlepos = -0.02
    else:
        sns.boxplot(x='classification_accuracy', y='names2', data=df_fig2, palette=palette_husl2, order=order)
        plt.xlabel('Classification accuracy')
        plt.xlim([0.49,1.01])
        xtitlepos = 0.48
    plt.ylabel('')

    # add extra text next to y-ticks that aren't there
    ticks_per_group = np.array([2+len(ranks), 2+2*len(ranks), 3+len(ranks), len(ranks), len(ranks)])
    additional_ticks = np.concatenate([[0],np.cumsum(ticks_per_group+2)])
    # np.concatenate([np.arange(2+len(ranks)),np.arange(2+2*len(ranks))+7,np.arange(3+len(ranks))+17,np.arange(len(ranks))+25,np.arange(len(ranks))+30])
    ticks_list = [np.arange(ticks_per_group[i]) + additional_ticks[i] for i in range(len(ticks_per_group))]
    ticks_list = np.concatenate(ticks_list)
    # print(ticks_list)
    # print(np.concatenate([np.arange(2+len(ranks)),np.arange(2+2*len(ranks))+7,np.arange(3+len(ranks))+17,np.arange(len(ranks))+25,np.arange(len(ranks))+30]))
    # additional_ticks = [0, 7, 17, 25, 30]
    # plt.yticks(np.concatenate([np.arange(2+len(ranks)),np.arange(2+2*len(ranks))+7,np.arange(3+len(ranks))+17,np.arange(len(ranks))+25,np.arange(len(ranks))+30]),fontsize=8)

    if len(ranks)==3:
        ytitlepos = [-0.7,6.3,16.3,24.3,29.3]
        plt.yticks(ticks_list, fontsize=8)
    elif len(ranks)==5:
        ytitlepos = [-0.7,8.3,22.3,32.3,39.3]
        plt.yticks(ticks_list, fontsize=7)
        plt.ylim([46,-2])
    elif len(ranks)==6:
        ytitlepos = [-0.7,9.3,25.3,36.3,44.3]
        plt.yticks(ticks_list, fontsize=6)
        plt.ylim([51,-2])
        
    plt.text(xtitlepos, ytitlepos[0], 'Complex-valued phase coupling', fontsize=8,fontweight='bold', ha='right')
    plt.text(xtitlepos, ytitlepos[1], 'Cosine phase coupling', fontsize=8,fontweight='bold', ha='right')
    plt.text(xtitlepos, ytitlepos[2], 'LEiDA (reduced cosine phase coupling)', fontsize=8,fontweight='bold', ha='right')
    plt.text(xtitlepos, ytitlepos[3], 'Amplitude coupling', fontsize=8,fontweight='bold', ha='right')
    plt.text(xtitlepos, ytitlepos[4], 'Phase-amplitude coupling', fontsize=8,fontweight='bold', ha='right')

    # change the line styles
    styles = ['-']+['-']*len(ranks)+['--']+[':']+['-']*len(ranks)+['-']*len(ranks)+['--']*2+['-']+['-']*len(ranks)+['--']*2+['-']*len(ranks)+['-']*len(ranks)
    #repeat every element of styles six times
    styles2 = [item for item in styles for i in range(5)]
    l = 0
    for i,artist in enumerate(plt.gca().get_children()):
        if isinstance(artist, plt.Line2D):
            #if linestyle is not none
            if artist.get_linestyle() != 'None':
                artist.set_linestyle(styles2[l])
                l+=1
        # print(l)
    # plt.savefig(savename, bbox_inches='tight', dpi=300)
    return fig

def horizontal_boxplot_revision(df_fig,type=1,ranks=[1,10,25]):
    order = [*['Mixture: Complex ACG rank='+str(rank) for rank in ranks],
        'K-means: Complex diametrical','space1','space2',
        *['Mixture: Gaussian rank='+str(rank) for rank in ranks],'space3','space4',
        *['Mixture: Complex Gaussian rank='+str(rank) for rank in ranks],'space5','space6',
        'K-means: Diametrical','K-means: Least squares (sign-flip)']
    palette_husl = sns.color_palette("husl", n_colors=11, desat=1)
    palette_husl2 = [palette_husl[1]]*len(ranks)+[palette_husl[2]]+[palette_husl[-1]]*2+[(0.5,0.5,0.5)]*len(ranks)+[palette_husl[-1]]*2+[(0.3,0.3,0.3)]*len(ranks)+[palette_husl[-1]]*2 + [palette_husl[3]] + [palette_husl[4]]# df_fig = df[df['Set']=='Out-of-sample test']
    for i in range(1,7):
        df_fig2 = pd.concat([df_fig,pd.DataFrame({'NMI':[np.nan],'names2':['space'+str(i)]}, index=[0])], ignore_index=True)
    fig = plt.figure(figsize=(7,5))
    if type == 1:
        sns.boxplot(x='NMI', y='names2', data=df_fig2, palette=palette_husl2, order=order)
        plt.xlabel('Normalized mutual information')
        plt.xlim([-0.01,1.01])
        xtitlepos = -0.02
    else:
        sns.boxplot(x='classification_accuracy', y='names2', data=df_fig2, palette=palette_husl2, order=order)
        plt.xlabel('Classification accuracy')
        plt.xlim([0.49,1.01])
        xtitlepos = 0.48
    plt.ylabel('')

    # add extra text next to y-ticks that aren't there
    ticks_per_group = np.array([1+len(ranks), len(ranks), len(ranks),2])
    additional_ticks = np.concatenate([[0],np.cumsum(ticks_per_group+2)])
    # np.concatenate([np.arange(2+len(ranks)),np.arange(2+2*len(ranks))+7,np.arange(3+len(ranks))+17,np.arange(len(ranks))+25,np.arange(len(ranks))+30])
    ticks_list = [np.arange(ticks_per_group[i]) + additional_ticks[i] for i in range(len(ticks_per_group))]
    ticks_list = np.concatenate(ticks_list)
    # print(ticks_list)
    # print(np.concatenate([np.arange(2+len(ranks)),np.arange(2+2*len(ranks))+7,np.arange(3+len(ranks))+17,np.arange(len(ranks))+25,np.arange(len(ranks))+30]))
    # additional_ticks = [0, 7, 17, 25, 30]
    # plt.yticks(np.concatenate([np.arange(2+len(ranks)),np.arange(2+2*len(ranks))+7,np.arange(3+len(ranks))+17,np.arange(len(ranks))+25,np.arange(len(ranks))+30]),fontsize=8)

    if len(ranks)==3:
        ytitlepos = [-0.7,5.3,8.3,11.3]
        plt.yticks(ticks_list, fontsize=8)
        plt.ylim([27,-2])
    elif len(ranks)==5:
        ytitlepos = [-0.7,7.3,14.3,21.3]
        plt.yticks(ticks_list, fontsize=7)
        plt.ylim([24,-2])
    elif len(ranks)==6:
        ytitlepos = [-0.7,8.3,12.3,20.3]
        plt.yticks(ticks_list, fontsize=6)
        plt.ylim([22,-2])
        
    plt.text(xtitlepos, ytitlepos[0], 'Complex-valued phase coupling', fontsize=8,fontweight='bold', ha='right')
    # plt.text(xtitlepos, ytitlepos[1], 'Cosine phase coupling', fontsize=8,fontweight='bold', ha='right')
    plt.text(xtitlepos, ytitlepos[1], 'Amplitude coupling', fontsize=8,fontweight='bold', ha='right')
    plt.text(xtitlepos, ytitlepos[2], 'Phase-amplitude coupling', fontsize=8,fontweight='bold', ha='right')
    plt.text(xtitlepos, ytitlepos[3], 'LEiDA (reduced cosine phase coupling)', fontsize=8,fontweight='bold', ha='right')

    # change the line styles
    styles = ['-']*len(ranks)+['--']+['-']*len(ranks)+['-']*len(ranks)+['--']*2
    #repeat every element of styles six times
    styles2 = [item for item in styles for i in range(5)]
    l = 0
    for i,artist in enumerate(plt.gca().get_children()):
        if isinstance(artist, plt.Line2D):
            #if linestyle is not none
            if artist.get_linestyle() != 'None':
                artist.set_linestyle(styles2[l])
                l+=1
        # print(l)
    # plt.savefig(savename, bbox_inches='tight', dpi=300)
    return fig

def run_phaserando(data_train,data_test1,data_test2,K,P,df,options,params=None,suppress_output=False,inner=None,p=116):
    params,train_posterior,loglik_curve = train_model(data_train,K=K,options=options,suppress_output=suppress_output,samples_per_sequence=1200,params=params)
    test1_loglik,test1_posterior,_ = test_model(data_test=data_test1,params=params,K=K,options=options,samples_per_sequence=1200)
    test2_loglik,test2_posterior,_ = test_model(data_test=data_test2,params=params,K=K,options=options,samples_per_sequence=1200)
    train_NMI = calc_NMI(P,np.double(np.array(train_posterior)))
    test1_NMI = calc_NMI(P,np.double(np.array(test1_posterior)))
    test2_NMI = calc_NMI(P,np.double(np.array(test2_posterior)))
    entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),'K':K,'p':p,'rank':options['rank'],'inner':inner,'iter':len(loglik_curve),
            'train_loglik':loglik_curve[-1],'test1_loglik':test1_loglik,'test2_loglik':test2_loglik,
            'train_NMI':train_NMI,'test1_NMI':test1_NMI,'test2_NMI':test2_NMI}
    df = pd.concat([df,pd.DataFrame([entry])],ignore_index=True)
    df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
    return params,df



# Paired Hotelling T² test for complex scalars (Python)
# This code computes the paired Hotelling T² statistic (one-sample Hotelling on paired differences),
# converts it to an F-statistic, and returns the analytic p-value.
#
# Requirements: numpy, scipy
# Usage: call hotelling_paired(z1, z2, mu0=0+0j)
#
# Returns: dict with T2, F, pval, xbar (2-vector), S (2x2 covariance)

def hotelling_paired(z1, z2, mu0=0+0j, regularize=True, reg_tol=1e-8, is_real=False):
    """
    Paired Hotelling T^2 test for complex scalar pairs.
    
    Null hypothesis: mean(z1 - z2) = mu0 (complex scalar)
    
    Parameters
    ----------
    z1, z2 : array_like, shape (n,)
        Arrays of complex observations (paired).
    mu0 : complex, optional
        Hypothesized mean difference (default 0+0j).
    regularize : bool, optional
        If True, automatically regularize the sample covariance when it's nearly singular.
    reg_tol : float, optional
        Relative tolerance for regularization (small positive number).
    
    Returns
    -------
    result : dict
        {
          'T2': Hotelling T^2 statistic,
          'F': corresponding F statistic,
          'pval': right-tail p-value from F_{p, n-p},
          'xbar': sample mean vector (2,),
          'S': sample covariance matrix (2,2),
          'n': sample size,
          'p': dimension (2)
        }
    """
    z1 = np.asarray(z1)
    z2 = np.asarray(z2)
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same shape.")
    if z1.ndim != 1:
        raise ValueError("z1 and z2 must be 1-D arrays of complex scalars.")
    if not is_real:
        if not np.iscomplexobj(z1) or not np.iscomplexobj(z2):
            raise ValueError("z1 and z2 must be arrays of complex numbers.")
    
    d = z1 - z2  # paired differences (complex)

    if is_real:
        # revert to a t-test of the real parts only
        from scipy.stats import ttest_rel
        t_stat, p_values = ttest_rel(z1.real, z2.real, axis=0, nan_policy='propagate')
        return {'T2': None,'F': None,'pval': p_values,'xbar': np.array([d.real.mean(), 0.0]),'S': None,'n': d.size,'p': 1}


    n = d.size
    p = 2  # real dimension after stacking real+imag

    # form 2D real vectors of (Re(d - mu0), Im(d - mu0))
    d_centered = d - mu0
    X = np.column_stack((d_centered.real, d_centered.imag))  # shape (n,2)
    
    xbar = X.mean(axis=0)  # sample mean (2,)
    # sample covariance with ddof=1
    S = np.cov(X, rowvar=False, ddof=1)
    
    # check invertibility / rank
    rankS = np.linalg.matrix_rank(S)
    if rankS < p:
        msg = f"Sample covariance is rank {rankS} < {p}; result would be unstable."
        if not regularize:
            raise np.linalg.LinAlgError(msg)
        # regularize: add small multiple of identity proportional to trace(S)
        traceS = np.trace(S)
        # fallback if trace is zero
        base = traceS if traceS > 0 else 1.0
        eps = reg_tol * base
        S = S + eps * np.eye(p)
        warnings.warn(msg + f" Regularizing with eps={eps:.2e}.", UserWarning)
    
    # compute T^2 = n * xbar' S^{-1} xbar
    try:
        Sinv_x = np.linalg.solve(S, xbar)  # solve S * y = xbar
        T2 = n * float(xbar.dot(Sinv_x))
    except np.linalg.LinAlgError:
        # as a last resort, use pseudo-inverse (shouldn't be needed with regularization)
        Sinv = np.linalg.pinv(S)
        T2 = n * float(xbar.dot(Sinv.dot(xbar)))
        warnings.warn("Used pseudo-inverse for S.", UserWarning)
    
    # convert to F
    Fnum = (n - p) / (p * (n - 1)) if n > p else np.nan
    Fstat = Fnum * T2
    # p-value (right tail)
    if not np.isnan(Fstat):
        pval = 1.0 - f.cdf(Fstat, dfn=p, dfd=(n - p))
    else:
        pval = np.nan
    
    return {
        'T2': T2,
        'F': Fstat,
        'pval': pval,
        'xbar': xbar,
        'S': S,
        'n': n,
        'p': p
    }


from scipy.optimize import linear_sum_assignment
def posterior_to_accuracy(
        posterior_train,
        posterior_test,
        cov_or_ts='cov'
):
    if cov_or_ts == 'ts':
        num_points = np.array([176,253,316,284,232,274,405])
    else:
        num_points = np.array([1,1,1,1,1,1,1])  # for cov
    K = len(num_points)
    total_points = np.sum(num_points)

    # cumulative indices
    cumsum_num_points = np.concatenate([[0], np.cumsum(num_points)]).astype(int)

    # true labels
    true_labels = np.zeros(total_points, dtype=int)
    for i in range(K):
        true_labels[cumsum_num_points[i]:cumsum_num_points[i+1]] = i

    n_subjects = 99
    n_train_subjects = 155

    accs = np.zeros((n_subjects))
    accs_ensemble = np.zeros((n_subjects))

    # average training posterior across subjects
    avg_train_posterior = np.zeros((K, total_points))
    for sub in range(n_train_subjects):
        avg_train_posterior += posterior_train[:, sub*total_points:(sub+1)*total_points] / n_train_subjects

    # normalize + binarize
    norm_avg_train_posterior = avg_train_posterior / np.sum(
        avg_train_posterior, axis=0, keepdims=True
    )
    bin_norm_avg_train_posterior = np.argmax(norm_avg_train_posterior, axis=0)

    # compute task–component accuracy matrix
    avg_accs = np.zeros((K, K))
    for k1 in range(K):          # true task
        for k2 in range(K):      # component
            avg_accs[k1, k2] = np.sum(
                (true_labels == k1) & (bin_norm_avg_train_posterior == k2)
            ) / np.sum(true_labels == k1)

    # Hungarian matching
    _, order = linear_sum_assignment(-avg_accs)

    for sub in range(n_subjects):
        test_posterior_sub = posterior_test[
            order, sub*total_points:(sub+1)*total_points
        ]

        # point-wise prediction
        tmp = np.argmax(test_posterior_sub, axis=0)

        accs[sub] = np.mean(tmp == true_labels)

        # ensemble accuracy
        if cov_or_ts == 'ts':
            for scan in range(K):
                tmp2 = test_posterior_sub[
                    :, cumsum_num_points[scan]:cumsum_num_points[scan+1]
                ].mean(axis=1)
                most_often_occurring_label = np.argmax(tmp2)
                accs_ensemble[sub] += (most_often_occurring_label == scan) / K

    return accs, accs_ensemble