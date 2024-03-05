import numpy as np
from tqdm import tqdm
import h5py as h5
from src.DMM_EM.mixture_EM_loop import mixture_EM_loop
from src.DMM_EM.WatsonEM import Watson as Watson_EM
from src.DMM_EM.ACGEM import ACG as ACG_EM
from src.DMM_EM.MACGEM import MACG as MACG_EM

from src.DMM_pytorch.mixture_torch_loop import mixture_torch_loop
from src.DMM_pytorch.WatsonPyTorch import Watson as Watson_torch
from src.DMM_pytorch.ACGPyTorch import ACG as ACG_torch
from src.DMM_pytorch.MACGPyTorch import MACG as MACG_torch

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
    if options['LR']==0:
        data_train = np.ascontiguousarray(data_train)
        data_test = np.ascontiguousarray(data_test)
    return data_train,data_test

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
    if options['modelname']=='Watson' or options['modelname']=='ACG' or options['modelname']=='ACG_lowrank':
        num_eigs=1
        data_train_all = np.zeros((1200*len(subjectlist),num_rois))
        data_test_all = np.zeros((1200*len(subjectlist),num_rois))
    elif options['modelname']=='MACG' or options['modelname']=='MACG_lowrank':
        num_eigs=2    
        data_train_all = np.zeros((1200*len(subjectlist),num_rois,num_eigs))
        data_test_all = np.zeros((1200*len(subjectlist),num_rois,num_eigs))
    else:
        raise ValueError('Invalid modelname')

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

    if options['LR']==0:
        data_train_all = np.ascontiguousarray(data_train_all)
        data_test_all = np.ascontiguousarray(data_test_all)

    return data_train_all,data_test_all
# def load_data(options,p=3,K=2):
#     import h5py

#     if options['data_type']=='fMRI_SchaeferTian454':
#         loc = 'data/processed/fMRI_SchaeferTian454_RL'
#     elif options['data_type']=='fMRI_full':
#         loc = 'data/processed/fMRI_full_RL'
#     elif options['data_type']=='fMRI_SchaeferTian116':
#         loc = 'data/processed/fMRI_SchaeferTian116_RL'
#     elif options['data_type']=='fMRI_SchaeferTian116_GSR':
#         loc = 'data/processed/fMRI_SchaeferTian116_GSR_RL'
#     else:
#         raise ValueError('Invalid data_type')

#     if options['modelname']=='Watson' or options['modelname']=='ACG' or options['modelname']=='ACG_lowrank':
#         num_eigs=1
#     elif options['modelname']=='MACG' or options['modelname']=='MACG_lowrank':
#         num_eigs=2
    
#     data_train_tmp = np.array(h5py.File(loc+'2.h5', 'r')['Dataset'][:,:options['num_subjects']*1200*2]).T
#     data_test_tmp = np.array(h5py.File(loc+'1.h5', 'r')['Dataset'][:,:options['num_subjects']*1200*2]).T
#     data_test2_tmp = np.array(h5py.File(loc+'2.h5', 'r')['Dataset'][:,options['num_subjects']*1200*2:options['num_subjects']*1200*4]).T
#     if num_eigs==1:
#         data_train = data_train_tmp[np.arange(options['num_subjects']*1200*2,step=2),:]
#         data_test = data_test_tmp[np.arange(options['num_subjects']*1200*2,step=2),:]
#         data_test2 = data_test2_tmp[np.arange(options['num_subjects']*1200*2,step=2),:]
#     elif num_eigs == 2:
#         p = data_train_tmp.shape[1]
#         data_train = np.zeros((options['num_subjects']*1200,p,2))
#         data_train[:,:,0] = data_train_tmp[np.arange(options['num_subjects']*1200*2,step=2),:]
#         data_train[:,:,1] = data_train_tmp[np.arange(options['num_subjects']*1200*2,step=2)+1,:]
#         data_test = np.zeros((options['num_subjects']*1200,p,2))
#         data_test[:,:,0] = data_test_tmp[np.arange(options['num_subjects']*1200*2,step=2),:]
#         data_test[:,:,1] = data_test_tmp[np.arange(options['num_subjects']*1200*2,step=2)+1,:]
#         data_test2 = np.zeros((options['num_subjects']*1200,p,2))
#         data_test2[:,:,0] = data_test2_tmp[np.arange(options['num_subjects']*1200*2,step=2),:]
#         data_test2[:,:,1] = data_test2_tmp[np.arange(options['num_subjects']*1200*2,step=2)+1,:]

#     if options['LR']!=0:
#         data_train = torch.tensor(data_train)
#         data_test = torch.tensor(data_test)
#         data_test2 = torch.tensor(data_test2)
#     return data_train,data_test,data_test2

def train_model(data_train,K,options,params=None,suppress_output=False):

    p = data_train.shape[1]
    if options['ACG_rank']=='fullrank':
        rank=0
    elif options['ACG_rank']=='lowrank': #assume full rank in lowrank setting
        rank=p
    else: 
        rank=options['ACG_rank']
    if options['modelname'] == 'Watson':
        if options['LR']==0:
            model = Watson_EM(K=K,p=p,params=params)
        else:
            model = Watson_torch(K=K,p=p,params=params,HMM=options['HMM'])
    elif options['modelname'] == 'ACG':
        if options['LR']==0:
            model = ACG_EM(K=K,p=p,rank=rank,params=params)
        else:
            model = ACG_torch(K=K,p=p,rank=rank,params=params,HMM=options['HMM']) #cholesky formulation when full rank       
    elif options['modelname'] == 'MACG':
        if options['LR']==0:
            model = MACG_EM(K=K,p=p,q=2,rank=rank,params=params)
        else:
            model = MACG_torch(K=K,p=p,q=2,rank=rank,params=params,HMM=options['HMM']) 
        
    if options['LR']==0: #EM
        params,posterior,loglik = mixture_EM_loop(model,data_train,tol=options['tol'],max_iter=options['max_iter'],
                                                num_repl=options['num_repl_inner'],init=options['init'],
                                                suppress_output=suppress_output,threads=options['threads'])
    else:
        params,posterior,loglik = mixture_torch_loop(model,data_train,tol=options['tol'],max_iter=options['max_iter'],
                                        num_repl=options['num_repl_inner'],init=options['init'],LR=options['LR'],
                                        suppress_output=suppress_output,threads=options['threads'])
    
    return params,posterior,loglik
    
def test_model(data_test,params,K,options):
    p = data_test.shape[1]
    if options['ACG_rank']=='fullrank':
        rank=0
    elif options['ACG_rank']=='lowrank': #assume full rank in lowrank setting
        rank=p
    else: 
        rank=options['ACG_rank']
    if options['LR'] == 0:
        if options['modelname'] == 'Watson':    
            model = Watson_EM(K=K,p=p,params=params)
        elif options['modelname'] == 'ACG':
            model = ACG_EM(K=K,p=p,rank=rank,params=params)
        elif options['modelname'] == 'MACG':
            model = MACG_EM(K=K,p=p,q=2,rank=rank,params=params)
    else:
        if options['modelname'] == 'Watson':
            model = Watson_torch(K=K,p=p,params=params)
        elif options['modelname'] == 'ACG':
            model = ACG_torch(K=K,p=p,rank=rank,params=params) 
        elif options['modelname'] == 'MACG':
            model = MACG_torch(K=K,p=p,q=2,rank=rank,params=params) 
    test_loglik = model.test_log_likelihood(X=data_test)  
    test_posterior = model.posterior(X=data_test)
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
    ind=np.where(PXY>0)
    MI=np.sum(PXY[ind]*np.log(PXY[ind]/PXPY[ind]))
    return MI

def calc_NMI(Z1,Z2):
    NMI = (2*calc_MI(Z1,Z2))/(calc_MI(Z1,Z1)+calc_MI(Z2,Z2))
    return NMI

# def parse_input_args(args):
#     options = {}
#     options['modelname'] = args[1]
#     options['LR'] = float(args[2])
#     options['init'] = args[3]
#     if len(args)>4:
#         options['GSR'] = int(args[4])
#     else:
#         options['GSR'] = None
#     return options


