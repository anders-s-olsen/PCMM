import numpy as np
# from tqdm import tqdm
from src.DMM_EM.mixture_EM_loop import mixture_EM_loop
from src.DMM_EM.WatsonEM import Watson as Watson_EM
from src.DMM_EM.ACGEM import ACG as ACG_EM
from src.DMM_EM.MACGEM import MACG as MACG_EM
from src.DMM_EM.SingularWishartEM import SingularWishart as SingularWishart_EM

from src.DMM_pytorch.mixture_torch_loop import mixture_torch_loop
from src.DMM_pytorch.WatsonPyTorch import Watson as Watson_torch
from src.DMM_pytorch.ACGPyTorch import ACG as ACG_torch
from src.DMM_pytorch.MACGPyTorch import MACG as MACG_torch
# from src.DMM_pytorch.MVGPyTorch import MVG as MVG_torch
from src.DMM_pytorch.SingularWishartPyTorch import SingularWishart as SingularWishart_torch

def load_synthetic_data(options,p,K):
    if options['modelname']=='Watson' or options['modelname']=='ACG' or options['modelname']=='ACG_lowrank':
        num_eigs=1
    elif options['modelname']=='MACG' or options['modelname']=='MACG_lowrank':
        num_eigs=2
    if num_eigs==1:
        data_train = np.loadtxt('data/synthetic/synth_data_ACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
        data_test = np.loadtxt('data/synthetic/synth_data_ACG_p'+str(p)+'K'+str(K)+'_2.csv',delimiter=',')
    elif num_eigs==2:
        data_train = np.zeros((10000,p,2))
        data_train_tmp = np.loadtxt('data/synthetic/synth_data_MACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
        data_train[:,:,0] = data_train_tmp[np.arange(20000,step=2),:]
        data_train[:,:,1] = data_train_tmp[np.arange(20000,step=2)+1,:]
        data_test = np.zeros((10000,p,2))
        data_test_tmp = np.loadtxt('data/synthetic/synth_data_MACG_p'+str(p)+'K'+str(K)+'_2.csv',delimiter=',')
        data_test[:,:,0] = data_test_tmp[np.arange(20000,step=2),:]
        data_test[:,:,1] = data_test_tmp[np.arange(20000,step=2)+1,:]
    # if options['LR']==0:
    #     data_train = np.ascontiguousarray(data_train)
    #     data_test = np.ascontiguousarray(data_test)
    return data_train,data_test

def train_model(data_train,L_train,K,options,params=None,suppress_output=False,samples_per_sequence=0):

    p = data_train.shape[1]
    if options['rank']=='fullrank':
        rank=0
    elif options['rank']=='lowrank': #assume full rank in lowrank setting
        rank=p
    else: 
        rank=options['rank']
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
    else:
        raise ValueError("Problem")
        
    if options['LR']==0: #EM
        params,posterior,loglik = mixture_EM_loop(model,data_train,L_train,tol=options['tol'],max_iter=options['max_iter'],
                                                num_repl=options['num_repl_inner'],init=options['init'],
                                                suppress_output=suppress_output)
    else:
        params,posterior,loglik = mixture_torch_loop(model,data_train,L_train,tol=options['tol'],max_iter=options['max_iter'],
                                        num_repl=options['num_repl_inner'],init=options['init'],LR=options['LR'],
                                        suppress_output=suppress_output,threads=options['threads'])
    return params,posterior,loglik
    
def test_model(data_test,L_test,params,K,options):
    p = data_test.shape[1]
    if options['rank']=='fullrank':
        rank=0
    elif options['rank']=='lowrank': #assume full rank in lowrank setting
        rank=p
    else: 
        rank=options['rank']
    if options['LR'] == 0:
        if options['modelname'] == 'Watson':    
            model = Watson_EM(K=K,p=p,params=params)
            X = data_test
        elif options['modelname'] == 'Complex_Watson':
            model = Watson_EM(K=K,p=p,complex=True,params=params)
            X = data_test
        elif options['modelname'] == 'ACG':
            model = ACG_EM(K=K,p=p,rank=rank,params=params)
            X = data_test
        elif options['modelname'] == 'Complex_ACG':
            model = ACG_EM(K=K,p=p,complex=True,rank=rank,params=params)
            X = data_test
        elif options['modelname'] == 'MACG':
            model = MACG_EM(K=K,p=p,q=2,rank=rank,params=params)
            X = data_test
        elif options['modelname'] == 'SingularWishart':
            model = SingularWishart_EM(K=K,p=p,q=2,rank=rank,params=params)
            X = data_test*np.sqrt(L_test[:,None,:])
        else:
            raise ValueError("Problem, modelname:",options['modelname'])
    else:
        if options['modelname'] == 'Watson':
            model = Watson_torch(K=K,p=p,params=params)
            X = data_test
        elif options['modelname'] == 'Complex_Watson':
            model = Watson_torch(K=K,p=p,complex=True,params=params)
            X = data_test
        elif options['modelname'] == 'ACG':
            model = ACG_torch(K=K,p=p,rank=rank,params=params) 
            X = data_test
        elif options['modelname'] == 'Complex_ACG':
            model = ACG_torch(K=K,p=p,complex=True,rank=rank,params=params) 
            X = data_test
        elif options['modelname'] == 'MACG':
            model = MACG_torch(K=K,p=p,q=2,rank=rank,params=params) 
            X = data_test
        elif options['modelname'] == 'SingularWishart':
            model = SingularWishart_torch(K=K,p=p,q=2,rank=rank,params=params)
            X = data_test*np.sqrt(L_test[:,None,:])
        else:
            raise ValueError("Problem, modelname:",options['modelname'])
    test_loglik = model.test_log_likelihood(X=X)  
    test_posterior = model.posterior(X=X)
    return test_loglik,test_posterior

def run_model_reps_and_save_logliks(data_train,data_test,K,options,data_test2=None):
    params,train_posterior,loglik_curve = train_model(data_train=data_train,K=K,options=options)
    test_loglik,test_posterior = test_model(data_test=data_test,params=params,K=K,options=options)
    if data_test2 is not None:
        test_loglik2,_ = test_model(data_test=data_test2,params=params,K=K,options=options)
    else:
        test_loglik2 = None
    
    return params,loglik_curve,test_loglik,test_loglik2,train_posterior,test_posterior
        
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

def make_true_mat(num_subs=10):
    rows = []
    for _ in range(num_subs):
        row = np.zeros((5,1200),dtype=bool)
        num_samples = 240
        row[0,num_samples*0:num_samples*1] = True
        row[1,num_samples*1:num_samples*2] = True
        row[2,num_samples*2:num_samples*3] = True
        row[3,num_samples*3:num_samples*4] = True
        row[4,num_samples*4:num_samples*5] = True
        rows.append(row)
    return np.hstack(rows)