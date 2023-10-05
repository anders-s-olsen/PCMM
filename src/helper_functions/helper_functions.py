import numpy as np
import h5py
import torch
from src.models_python.ACGMixtureEM import ACG as ACG_EM
from src.models_python.ACGLowrankEM import ACG as ACG_lowrank_EM
from src.models_python.MACGMixtureEM import MACG as MACG_EM
from src.models_python.MACGLowrankEM import MACG as MACG_lowrank_EM
from src.models_python.WatsonMixtureEM import Watson as Watson_EM
from src.models_python.mixture_EM_loop import mixture_EM_loop
from src.models_pytorch.ACG_lowrank_torch import ACG as ACG_torch
from src.models_pytorch.MACG_lowrank_torch import MACG as MACG_torch
from src.models_pytorch.Watson_torch import Watson as Watson_torch
from src.models_pytorch.mixture_torch_loop import mixture_torch_loop

def load_data(type,num_subjects=200,num_eigs=1,LR=0,p=3,K=2):

    if type=='fMRI_SchaeferTian454':
        loc = 'data/processed/fMRI_SchaeferTian454_RL'
    elif type=='fMRI_full':
        loc = 'data/processed/fMRI_full_RL'
    elif type=='fMRI_SchaeferTian116':
        loc = 'data/processed/fMRI_SchaeferTian116_RL'
    elif type=='fMRI_SchaeferTian116_GSR':
        loc = 'data/processed/fMRI_SchaeferTian116_GSR_RL'
    elif type=='synth' or type=='synthetic':
        loc = 'data/synthetic/synth_data_'
    
    if type!='synth' and type!='synthetic':
        data_train_tmp = np.array(h5py.File(loc+'2.h5', 'r')['Dataset'][:,:num_subjects*1200*2]).T
        data_test_tmp = np.array(h5py.File(loc+'1.h5', 'r')['Dataset'][:,:num_subjects*1200*2]).T
        data_test2_tmp = np.array(h5py.File(loc+'2.h5', 'r')['Dataset'][:,num_subjects*1200*2:num_subjects*1200*4]).T
        if num_eigs==1:
            data_train = data_train_tmp[np.arange(num_subjects*1200*2,step=2),:]
            data_test = data_test_tmp[np.arange(num_subjects*1200*2,step=2),:]
            data_test2 = data_test2_tmp[np.arange(num_subjects*1200*2,step=2),:]
        elif num_eigs == 2:
            p = data_train_tmp.shape[1]
            data_train = np.zeros((num_subjects*1200,p,2))
            data_train[:,:,0] = data_train_tmp[np.arange(num_subjects*1200*2,step=2),:]
            data_train[:,:,1] = data_train_tmp[np.arange(num_subjects*1200*2,step=2)+1,:]
            data_test = np.zeros((num_subjects*1200,p,2))
            data_test[:,:,0] = data_test_tmp[np.arange(num_subjects*1200*2,step=2),:]
            data_test[:,:,1] = data_test_tmp[np.arange(num_subjects*1200*2,step=2)+1,:]
            data_test2 = np.zeros((num_subjects*1200,p,2))
            data_test2[:,:,0] = data_test2_tmp[np.arange(num_subjects*1200*2,step=2),:]
            data_test2[:,:,1] = data_test2_tmp[np.arange(num_subjects*1200*2,step=2)+1,:]
    elif type=='synth' or type=='synthetic':
        if num_eigs==1:
            data_train = np.loadtxt('data/synthetic/synth_data_ACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
            data_test = np.loadtxt('data/synthetic/synth_data_ACG_p'+str(p)+'K'+str(K)+'_2.csv',delimiter=',')
        elif num_eigs==2:
            data_train = np.zeros((1000,p,2))
            data_train_tmp = np.loadtxt('data/synthetic/synth_data_MACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
            data_train[:,:,0] = data_train_tmp[np.arange(2000,step=2),:]
            data_train[:,:,1] = data_train_tmp[np.arange(2000,step=2)+1,:]
            data_test = np.zeros((1000,p,2))
            data_test_tmp = np.loadtxt('data/synthetic/synth_data_MACG_p'+str(p)+'K'+str(K)+'_2.csv',delimiter=',')
            data_test[:,:,0] = data_test_tmp[np.arange(2000,step=2),:]
            data_test[:,:,1] = data_test_tmp[np.arange(2000,step=2)+1,:]
        data_test2 = np.zeros(1)

    if LR!=0:
        data_train = torch.tensor(data_train)
        data_test = torch.tensor(data_test)
        data_test2 = torch.tensor(data_test2)
    return data_train,data_test,data_test2

def train_model(modelname,K,data_train,rank,init,LR,num_repl_inner,num_iter,tol,params=None):
    p = data_train.shape[1]
    if modelname == 'Watson':
        if LR==0:
            model = Watson_EM(K=K,p=p,params=params)
        else:
            model = Watson_torch(K=K,p=p,params=params)
    elif modelname == 'ACG':
        if LR==0:
            model = ACG_EM(K=K,p=p,params=params)
        else:
            model = ACG_torch(K=K,p=p,rank=rank,params=params) #cholesky formulation when full rank
    elif modelname == 'ACG_lowrank':
        if LR==0:
            model = ACG_lowrank_EM(K=K,p=p,rank=rank,params=params)
        else:
            model = ACG_torch(K=K,p=p,rank=rank,params=params) #cholesky formulation when full rank            
    elif modelname == 'MACG':
        if LR==0:
            model = MACG_EM(K=K,p=p,q=2,params=params)
        else:
            model = MACG_torch(K=K,p=p,q=2,rank=rank,params=params)
    elif modelname == 'MACG_lowrank':
        if LR==0:
            model = MACG_lowrank_EM(K=K,p=p,q=2,rank=rank,params=params)
        else:
            model = MACG_torch(K=K,p=p,q=2,rank=rank,params=params)   
        
    
    if LR==0: #EM
        params,_,loglik,_ = mixture_EM_loop(model,data_train,tol=tol,max_iter=num_iter,
                                                num_repl=num_repl_inner,init=init)
    else:
        params,_,loglik,_ = mixture_torch_loop(model,data_train,tol=tol,max_iter=num_iter,
                                        num_repl=num_repl_inner,init=init,LR=LR)
    
    return params,loglik[-1],loglik
    
def test_model(modelname,K,data_test,params,LR,rank):
    p = data_test.shape[1]
    if LR == 0:
        if modelname == 'Watson':    
            model = Watson_EM(K=K,p=p,params=params)
        elif modelname == 'ACG':
            model = ACG_EM(K=K,p=p,params=params)
        elif modelname == 'ACG_lowrank':
            model = ACG_EM(K=K,p=p,rank=rank,params=params)
        elif modelname == 'MACG':
            model = MACG_EM(K=K,p=p,q=2,params=params)
        test_loglik = model.log_likelihood(X=data_test)
        params_transformed = model.get_params()
    else:
        Softmax = torch.nn.Softmax(dim=0)
        pi_soft = Softmax(params['pi'])
        if modelname == 'Watson':
            mu_norm = torch.nn.functional.normalize(params['mu'],dim=0)
            kappa = params['kappa']
            params_transformed={'mu':mu_norm,'kappa':kappa,'pi':pi_soft}
        else:
            M = params['M']
            params_transformed={'M':M,'pi':pi_soft}
        with torch.no_grad():
            if modelname == 'Watson':
                model = Watson_torch(K=K,p=p,params=params_transformed)
            elif modelname == 'ACG':
                model = ACG_torch(K=K,p=p,rank=rank,params=params_transformed) #cholesky formulation when full rank
            elif modelname == 'MACG':
                model = MACG_torch(K=K,p=p,q=2,rank=rank,params=params_transformed) #cholesky formulation when full rank
            test_loglik = model.test_log_likelihood(X=data_test)  
    return test_loglik,params_transformed