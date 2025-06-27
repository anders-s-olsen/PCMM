import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from PCMM.mixture_EM_loop import mixture_EM_loop
from PCMM.PCMMnumpy import Watson as Watson_numpy
from PCMM.PCMMnumpy import ACG as ACG_numpy
from PCMM.PCMMnumpy import MACG as MACG_numpy
from PCMM.PCMMnumpy import SingularWishart as SingularWishart_numpy
from PCMM.PCMMnumpy import Normal as Normal_numpy
from PCMM.phase_coherence_kmeans import *

from PCMM.mixture_torch_loop import mixture_torch_loop
from PCMM.PCMMtorch import Watson as Watson_torch
from PCMM.PCMMtorch import ACG as ACG_torch
from PCMM.PCMMtorch import MACG as MACG_torch
from PCMM.PCMMtorch import SingularWishart as SingularWishart_torch
from PCMM.PCMMtorch import Normal as Normal_torch

def train_model(data_train,K,options,params=None,suppress_output=False,samples_per_sequence=0):
    p = data_train.shape[1]
    if options['modelname'] in ['Watson','Complex_Watson','ACG','Complex_ACG','MACG','SingularWishart','Normal','Complex_Normal']:
        if options['rank']=='fullrank':
            rank=0
        elif options['rank']=='lowrank': #assume full rank in lowrank setting
            rank=p
        else: 
            rank=options['rank']
        if options['LR']!=0:
            # data_train = torch.tensor(data_train)
            data_train = torch.from_numpy(data_train)

    if options['modelname'] == 'Watson':
        if options['LR']==0:
            model = Watson_numpy(K=K,p=p,params=params)
        else:
            model = Watson_torch(K=K,p=p,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'Complex_Watson':
        if options['LR']==0:
            model = Watson_numpy(K=K,p=p,complex=True,params=params)
        else:
            model = Watson_torch(K=K,p=p,complex=True,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'ACG':
        if options['LR']==0:
            model = ACG_numpy(K=K,p=p,rank=rank,params=params)
        else:
            model = ACG_torch(K=K,p=p,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'Complex_ACG':
        if options['LR']==0:
            model = ACG_numpy(K=K,p=p,rank=rank,complex=True,params=params)
        else:
            model = ACG_torch(K=K,p=p,rank=rank,complex=True,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'MACG':
        if options['LR']==0:
            model = MACG_numpy(K=K,p=p,q=2,rank=rank,params=params)
        else:
            model = MACG_torch(K=K,p=p,q=2,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence) 
    elif options['modelname'] == 'SingularWishart':
        if options['LR']==0:
            model = SingularWishart_numpy(K=K,p=p,q=2,rank=rank,params=params)
        else:
            model = SingularWishart_torch(K=K,p=p,q=2,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence) 
    elif options['modelname'] == 'Normal':
        if options['LR']==0:
            model = Normal_numpy(K=K,p=p,rank=rank,params=params)
        else:
            model = Normal_torch(K=K,p=p,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'Complex_Normal':
        if options['LR']==0:
            model = Normal_numpy(K=K,p=p,rank=rank,params=params,complex=True)
        else:
            model = Normal_torch(K=K,p=p,rank=rank,params=params,complex=True,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'least_squares':
        C,labels,obj = least_squares_sign_flip(data_train,K=K,max_iter=options['max_iter'],num_repl=options['num_repl'],init=options['init'],tol=options['tol'])
        # X = data_train
        # X[(X>0).sum(axis=1)>p/2] = -X[(X>0).sum(axis=1)>p/2]
        # C,labels = kmeans2(X,k=K,minit=options['init'])
        labels = np.eye(K)[labels].T
        params = {'C':C}
        #euclidean distance
        # sim = -np.sum((X[:,None]-params['C'][None])**2,axis=-1)
        # obj = np.mean(np.max(sim,axis=1))
        return params,labels,obj
    elif options['modelname'] in ['diametrical','complex_diametrical']:
        C,labels,obj = diametrical_clustering(data_train,K=K,max_iter=options['max_iter'],num_repl=options['num_repl'],init=options['init'],tol=options['tol'])
        labels = np.eye(K)[labels].T
        params = {'C':C}
        return params,labels,obj
    elif options['modelname'] == 'grassmann':
        C,labels,obj = grassmann_clustering(data_train,K=K,max_iter=options['max_iter'],num_repl=options['num_repl'],init=options['init'],tol=options['tol'])
        labels = np.eye(K)[labels].T
        params = {'C':C}
        return params,labels,obj
    elif options['modelname'] == 'weighted_grassmann':
        C,labels,obj = weighted_grassmann_clustering(data_train,K=K,max_iter=options['max_iter'],num_repl=options['num_repl'],init=options['init'],tol=options['tol'])
        labels = np.eye(K)[labels].T
        params = {'C':C}
        return params,labels,obj
    else:
        raise ValueError("Problem")
        
    #if the element 'tol' doesn't exist in options, set it to default 1e-10
    if 'tol' not in options:
        options['tol'] = 1e-10
    if 'max_iter' not in options:
        options['max_iter'] = 100000
    if 'num_repl' not in options:
        options['num_repl'] = 1
    if 'init' not in options:
        raise ValueError('Please provide an initialization method')
    if 'LR' not in options:
        options['LR'] = 0
    if 'threads' not in options:
        options['threads'] = 8
    if 'decrease_lr_on_plateau' not in options:
        options['decrease_lr_on_plateau'] = False
    if 'num_comparison' not in options:
        options['num_comparison'] = 50

    if options['LR']==0: #EM
        params,posterior,loglik = mixture_EM_loop(model,data_train,tol=options['tol'],max_iter=options['max_iter'],
                                                num_repl=options['num_repl'],init=options['init'],
                                                suppress_output=suppress_output)
    else:
        params,posterior,loglik = mixture_torch_loop(model,data_train,tol=options['tol'],max_iter=options['max_iter'],
                                        num_repl=options['num_repl'],init=options['init'],LR=options['LR'],
                                        suppress_output=suppress_output,threads=options['threads'],decrease_lr_on_plateau=options['decrease_lr_on_plateau'],num_comparison=options['num_comparison'])
    return params,posterior,loglik
    
def test_model(data_test,params,K,options,samples_per_sequence=0):
    p = data_test.shape[1]
    # if rank is a key in options
    if 'rank' in options:
        if options['rank']=='fullrank':
            rank=0
        elif options['rank']=='lowrank': #assume full rank in lowrank setting
            rank=p
        else: 
            rank=options['rank']
    
    if options['modelname'] in ['Watson','Complex_Watson','ACG','Complex_ACG','MACG','SingularWishart','Normal','Complex_Normal']:
        if options['LR']!=0:
            # data_test = torch.tensor(data_test)
            data_test = torch.from_numpy(data_test)
            
        if options['modelname'] == 'Watson':    
            if options['LR']==0:
                model = Watson_numpy(K=K,p=p,params=params)
            else:
                model = Watson_torch(K=K,p=p,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'Complex_Watson':
            if options['LR']==0:
                model = Watson_numpy(K=K,p=p,complex=True,params=params)
            else:
                model = Watson_torch(K=K,p=p,complex=True,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'ACG':
            if options['LR']==0:
                model = ACG_numpy(K=K,p=p,rank=rank,params=params)
            else:
                model = ACG_torch(K=K,p=p,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'Complex_ACG':
            if options['LR']==0:
                model = ACG_numpy(K=K,p=p,complex=True,rank=rank,params=params)
            else:
                model = ACG_torch(K=K,p=p,complex=True,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'MACG':
            if options['LR']==0:
                model = MACG_numpy(K=K,p=p,q=2,rank=rank,params=params)
            else:
                model = MACG_torch(K=K,p=p,q=2,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'SingularWishart':
            if options['LR']==0:
                model = SingularWishart_numpy(K=K,p=p,q=2,rank=rank,params=params)
            else:
                model = SingularWishart_torch(K=K,p=p,q=2,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'Normal':
            if options['LR']==0:
                model = Normal_numpy(K=K,p=p,rank=rank,params=params)
            else:
                model = Normal_torch(K=K,p=p,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'Complex_Normal':
            if options['LR']==0:
                model = Normal_numpy(K=K,p=p,rank=rank,complex=True,params=params)
            else:
                model = Normal_torch(K=K,p=p,rank=rank,complex=True,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
        test_loglik, test_loglik_per_sample = model.test_log_likelihood(X=data_test)
        test_posterior = model.posterior(X=data_test)
    elif options['modelname'] in ['least_squares','diametrical','complex_diametrical','grassmann','weighted_grassmann']:
        if options['modelname'] == 'least_squares':
            #eucdliean distance
            X=data_test
            X[(X>0).sum(axis=1)>p/2] = -X[(X>0).sum(axis=1)>p/2]
            sim = -np.sum((X[:,None]-params['C'][None])**2,axis=-1)
        elif options['modelname'] in ['diametrical','complex_diametrical']:
            sim = np.abs(data_test@params['C'].conj().T)**2
        elif options['modelname'] == 'grassmann':
            sim = -1/np.sqrt(2)*(2*data_test.shape[2]-2*np.linalg.norm(np.swapaxes(data_test[:,None],-2,-1)@params['C'][None],axis=(-2,-1))**2)
        elif options['modelname'] == 'weighted_grassmann':
            C_weights = np.linalg.norm(params['C'],axis=1)**2
            L_test = np.linalg.norm(data_test,axis=1)**2
            B = np.swapaxes(data_test,-2,-1)[:,None]@(params['C'][None])
            sim = -1/np.sqrt(2)*(np.sum(L_test**2,axis=-1)[:,None]+np.sum(C_weights**2,axis=-1)[None]-2*np.linalg.norm(B,axis=(-2,-1))**2)#
        
        test_loglik = np.mean(np.max(sim,axis=1))
        test_loglik_per_sample = np.max(sim,axis=1)
        
        test_posterior = np.argmax(sim,axis=1) # assign each point to the cluster with the highest similarity
        test_posterior = np.eye(K)[test_posterior].T
    else:
        raise ValueError("Problem, modelname:",options['modelname'])
    
    return test_loglik.item(),test_posterior,test_loglik_per_sample