import numpy as np
from scipy.cluster.vq import kmeans2
import torch
import h5py as h5
# from tqdm import tqdm
from PCMM.PCMM_EM.mixture_EM_loop import mixture_EM_loop
from PCMM.PCMM_EM.WatsonEM import Watson as Watson_EM
from PCMM.PCMM_EM.ACGEM import ACG as ACG_EM
from PCMM.PCMM_EM.MACGEM import MACG as MACG_EM
from PCMM.PCMM_EM.SingularWishartEM import SingularWishart as SingularWishart_EM
from PCMM.PCMM_EM.NormalEM import Normal as Normal_EM
from PCMM.riemannian_clustering import *

from PCMM.PCMM_pytorch.mixture_torch_loop import mixture_torch_loop
from PCMM.PCMM_pytorch.WatsonPyTorch import Watson as Watson_torch
from PCMM.PCMM_pytorch.ACGPyTorch import ACG as ACG_torch
from PCMM.PCMM_pytorch.MACGPyTorch import MACG as MACG_torch
from PCMM.PCMM_pytorch.SingularWishartPyTorch import SingularWishart as SingularWishart_torch
from PCMM.PCMM_pytorch.NormalPyTorch import Normal as Normal_torch

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
            data_train = torch.tensor(data_train)

    if options['modelname'] == 'Watson':
        if options['LR']==0:
            model = Watson_EM(K=K,p=p,params=params)
        else:
            model = Watson_torch(K=K,p=p,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'Complex_Watson':
        if options['LR']==0:
            model = Watson_EM(K=K,p=p,complex=True,params=params)
        else:
            model = Watson_torch(K=K,p=p,complex=True,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'ACG':
        if options['LR']==0:
            model = ACG_EM(K=K,p=p,rank=rank,params=params)
        else:
            model = ACG_torch(K=K,p=p,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'Complex_ACG':
        if options['LR']==0:
            model = ACG_EM(K=K,p=p,rank=rank,complex=True,params=params)
        else:
            model = ACG_torch(K=K,p=p,rank=rank,complex=True,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'MACG':
        if options['LR']==0:
            model = MACG_EM(K=K,p=p,q=2,rank=rank,params=params)
        else:
            model = MACG_torch(K=K,p=p,q=2,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence) 
    elif options['modelname'] == 'SingularWishart':
        if options['LR']==0:
            model = SingularWishart_EM(K=K,p=p,q=2,rank=rank,params=params)
        else:
            model = SingularWishart_torch(K=K,p=p,q=2,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence) 
    elif options['modelname'] == 'Normal':
        if options['LR']==0:
            model = Normal_EM(K=K,p=p,rank=rank,params=params)
        else:
            model = Normal_torch(K=K,p=p,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'Complex_Normal':
        if options['LR']==0:
            model = Normal_EM(K=K,p=p,rank=rank,params=params,complex=True)
        else:
            model = Normal_torch(K=K,p=p,rank=rank,params=params,complex=True,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'euclidean':
        X = data_train
        X[(X>0).sum(axis=1)>p/2] = -X[(X>0).sum(axis=1)>p/2]
        C,labels = kmeans2(X,k=K,minit=options['init'])
        labels = np.eye(K)[labels].T
        params = {'C':C}
        #euclidean distance
        sim = -np.sum((X[:,None]-params['C'][None])**2,axis=-1)
        obj = np.mean(np.max(sim,axis=1))
        return params,labels,[obj]
    elif options['modelname'] in ['diametrical','complex_diametrical']:
        C,labels,obj = diametrical_clustering(data_train,K=K,max_iter=options['max_iter'],num_repl=1,init=options['init'],tol=options['tol'])
        labels = np.eye(K)[labels].T
        params = {'C':C}
        return params,labels,[obj]
    elif options['modelname'] == 'grassmann':
        C,labels,obj = grassmann_clustering(data_train,K=K,max_iter=options['max_iter'],num_repl=1,init=options['init'],tol=options['tol'])
        labels = np.eye(K)[labels].T
        params = {'C':C}
        return params,labels,[obj]
    elif options['modelname'] == 'weighted_grassmann':
        C,C_weights,labels,obj = weighted_grassmann_clustering(data_train,K=K,max_iter=options['max_iter'],num_repl=1,init=options['init'],tol=options['tol'])
        labels = np.eye(K)[labels].T
        params = {'C':C,'C_weights':C_weights}
        return params,labels,[obj]
    else:
        raise ValueError("Problem")
        
    #if the element 'tol' doesn't exist in options, set it to default 1e-10
    if 'tol' not in options:
        options['tol'] = 1e-10
    if 'max_iter' not in options:
        options['max_iter'] = 100000
    if 'num_repl_inner' not in options:
        options['num_repl_inner'] = 1


    if options['LR']==0: #EM
        params,posterior,loglik = mixture_EM_loop(model,data_train,tol=options['tol'],max_iter=options['max_iter'],
                                                num_repl=options['num_repl_inner'],init=options['init'],
                                                suppress_output=suppress_output)
    else:
        params,posterior,loglik = mixture_torch_loop(model,data_train,tol=options['tol'],max_iter=options['max_iter'],
                                        num_repl=options['num_repl_inner'],init=options['init'],LR=options['LR'],
                                        suppress_output=suppress_output,threads=options['threads'])
    return 
    
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
            data_test = torch.tensor(data_test)
            
        if options['modelname'] == 'Watson':    
            if options['LR']==0:
                model = Watson_EM(K=K,p=p,params=params)
            else:
                model = Watson_torch(K=K,p=p,params=params,samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'Complex_Watson':
            if options['LR']==0:
                model = Watson_EM(K=K,p=p,complex=True,params=params)
            else:
                model = Watson_torch(K=K,p=p,complex=True,params=params,samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'ACG':
            if options['LR']==0:
                model = ACG_EM(K=K,p=p,rank=rank,params=params)
            else:
                model = ACG_torch(K=K,p=p,rank=rank,params=params,samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'Complex_ACG':
            if options['LR']==0:
                model = ACG_EM(K=K,p=p,complex=True,rank=rank,params=params)
            else:
                model = ACG_torch(K=K,p=p,complex=True,rank=rank,params=params,samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'MACG':
            if options['LR']==0:
                model = MACG_EM(K=K,p=p,q=2,rank=rank,params=params)
            else:
                model = MACG_torch(K=K,p=p,q=2,rank=rank,params=params,samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'SingularWishart':
            if options['LR']==0:
                model = SingularWishart_EM(K=K,p=p,q=2,rank=rank,params=params)
            else:
                model = SingularWishart_torch(K=K,p=p,q=2,rank=rank,params=params,samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'Normal':
            if options['LR']==0:
                model = Normal_EM(K=K,p=p,rank=rank,params=params)
            else:
                model = Normal_torch(K=K,p=p,rank=rank,params=params,samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'Complex_Normal':
            if options['LR']==0:
                model = Normal_EM(K=K,p=p,rank=rank,params=params,complex=True)
            else:
                model = Normal_torch(K=K,p=p,rank=rank,params=params,complex=True,samples_per_sequence=samples_per_sequence)
        test_loglik, test_loglik_per_sample = model.test_log_likelihood(X=data_test)
        test_posterior = model.posterior(X=data_test)
    elif options['modelname'] in ['euclidean','diametrical','complex_diametrical','grassmann','weighted_grassmann']:
        if options['modelname'] == 'euclidean':
            #eucdliean distance
            X=data_test
            X[(X>0).sum(axis=1)>p/2] = -X[(X>0).sum(axis=1)>p/2]
            sim = -np.sum((X[:,None]-params['C'][None])**2,axis=-1)
        elif options['modelname'] in ['diametrical','complex_diametrical']:
            sim = np.abs(data_test@params['C'].conj())**2
        elif options['modelname'] == 'grassmann':
            sim = -1/np.sqrt(2)*(2*data_test.shape[2]-2*np.linalg.norm(np.swapaxes(data_test[:,None],-2,-1)@params['C'][None],axis=(-2,-1))**2)
        elif options['modelname'] == 'weighted_grassmann':
            L_test = np.linalg.norm(data_test,axis=1)**2
            B = np.swapaxes(data_test,-2,-1)[:,None]@((params['C']*np.sqrt(params['C_weights'])[:,None,:])[None])
            sim = -1/np.sqrt(2)*(np.sum(L_test**2,axis=-1)[:,None]+np.sum(params['C_weights']**2,axis=-1)[None]-2*np.linalg.norm(B,axis=(-2,-1))**2)#
        
        test_loglik = np.mean(np.max(sim,axis=1))
        test_loglik_per_sample = np.max(sim,axis=1)
        
        test_posterior = np.argmax(sim,axis=1) # assign each point to the cluster with the highest similarity
        test_posterior = np.eye(K)[test_posterior].T
    else:
        raise ValueError("Problem, modelname:",options['modelname'])
    
    return test_loglik,test_posterior,test_loglik_per_sample
        
def calc_MI(Z1,Z2):
    P=Z1@Z2.T
    PXY=P/np.sum(P)
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

def load_fMRI_data(data_file,options,only_some_points=False):
    assert options['modelname'] in ['Watson','ACG','MACG','SingularWishart','Complex_Watson','Complex_ACG','Normal','Complex_Normal','euclidean','diametrical','complex_diametrical','grassmann','weighted_grassmann']

    with h5.File(data_file,'r') as f:
        if options['modelname'] in ['Complex_Watson','Complex_ACG','complex_diametrical']:
            # complex normalized phase vectors
            data_train = f['U_complex_train'][:][:,:,0]
            data_test1 = f['U_complex_test1'][:][:,:,0]
            data_test2 = f['U_complex_test2'][:][:,:,0] 
        elif options['modelname'] in ['Watson','ACG','euclidean','diametrical']:
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
    if only_some_points:
        data_train = data_train[:1000]
        data_test1 = data_test1[:1000]
        data_test2 = data_test2[:1000]
    return data_train,data_test1,data_test2

def calc_ll_per_sub(train_loglik_per_sample,test1_loglik_per_sample,test2_loglik_per_sample):
    if train_loglik_per_sample.shape[0]==155:
        return train_loglik_per_sample,test1_loglik_per_sample,test2_loglik_per_sample
    num_samples_per_sub_train = train_loglik_per_sample.shape[0]//155
    num_samples_per_sub_test1 = test1_loglik_per_sample.shape[0]//155
    num_samples_per_sub_test2 = test2_loglik_per_sample.shape[0]//100

    train_loglik_per_sub = np.zeros(155)
    test1_loglik_per_sub = np.zeros(155)
    test2_loglik_per_sub = np.zeros(100)

    for i in range(155):
        train_loglik_per_sub[i] = np.mean(train_loglik_per_sample[i*num_samples_per_sub_train:(i+1)*num_samples_per_sub_train])
        test1_loglik_per_sub[i] = np.mean(test1_loglik_per_sample[i*num_samples_per_sub_test1:(i+1)*num_samples_per_sub_test1])
        if i<100:
            test2_loglik_per_sub[i] = np.mean(test2_loglik_per_sample[i*num_samples_per_sub_test2:(i+1)*num_samples_per_sub_test2])

    return train_loglik_per_sub,test1_loglik_per_sub,test2_loglik_per_sub