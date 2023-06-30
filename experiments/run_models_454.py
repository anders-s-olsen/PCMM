import h5py
import numpy as np
import torch
from src.models_python.ACGMixtureEM import ACG as ACG_EM
from src.models_python.WatsonMixtureEM import Watson as Watson_EM
from src.models_python.mixture_EM_loop import mixture_EM_loop
from src.models_pytorch.ACG_lowrank_torch import ACG as ACG_torch
from src.models_pytorch.Watson_torch import Watson as Watson_torch
from src.models_pytorch.mixture_torch_loop import mixture_torch_loop

torch.set_num_threads(16)
import sys
import os
os.environ["OMP_NUM_THREADS"] = '16'
import matplotlib.pyplot as plt
expname = '3d'


def run_experiment(exp):
    ### load data, only the first 200 subjects (each with 1200 data points) (not the same subjects in train/test)
    # data_train = np.array(h5py.File('data/processed/fMRI_atlas_RL2.h5', 'r')['Dataset'][:,:240000]).T
    # n,p = data_train.shape
    # print('Loaded training data')
    # data_test = np.array(h5py.File('data/processed/fMRI_atlas_RL1.h5', 'r')['Dataset'][:,:240000]).T
    # print('Loaded test data, beginning fit')

    data_train = np.loadtxt('data/synthetic/synth_data_ACG.csv',delimiter=',')
    data_test = np.loadtxt('data/synthetic/synth_data_ACG2.csv',delimiter=',')
    p=3

    num_repl = 3
    K = 2

    ### EM algorithms
    print('starting K='+str(K))
    if exp==0:
        model = Watson_EM(K=K,p=p)
        name='Watson_EM'
    elif exp==1:
        model = ACG_EM(K=K,p=p)
        name='ACG_EM'
    elif exp==2:
        model = Watson_torch(K=K,p=p)
        name='Watson_torch'
    elif exp==3:
        model = ACG_torch(K=K,p=p,rank=p) #cholesky formulation when full rank
        name='ACG_torch'

    if exp==0 or exp==1:
        params,_,loglik,num_iter = mixture_EM_loop(model,data_train,tol=1e-6,max_iter=100000,num_repl=num_repl,init='++')
    elif exp==2 or exp==3:
        params,_,loglik,num_iter = mixture_torch_loop(model,torch.tensor(data_train),tol=1e-6,max_iter=100000,num_repl=num_repl,init='++')
        Softmax = torch.nn.Softmax(dim=0)
        Softplus = torch.nn.Softplus(beta=20, threshold=1)
        params['pi'] = Softmax(params['pi'])
        if exp ==2:
            params['mu'] = torch.nn.functional.normalize(params['mu'],dim=0)
            params['kappa'] = Softplus(params['kappa'])
        elif exp == 3:
            params['Lambda'] = torch.zeros(K,p,p)
            for k in range(K):
                L_tri_inv = params['L_tri_inv'][k]
                params['Lambda'][k] = torch.linalg.inv(L_tri_inv@L_tri_inv.T)
                params['Lambda'][k] = p*params['Lambda'][k]/torch.trace(params['Lambda'][k])

    np.savetxt('experiments/outputs'+expname+'/'+name+'_'+expname+'_trainlikelihoodcurve_K='+str(K)+'.csv',np.array(loglik))
    np.savetxt('experiments/outputs'+expname+'/'+name+'_'+expname+'_pi_K='+str(K)+'.csv',params['pi'])
    if exp==0 or exp==2:
        np.savetxt('experiments/outputs'+expname+'/'+name+'_'+expname+'_mu_K='+str(K)+'.csv',params['mu'])
        np.savetxt('experiments/outputs'+expname+'/'+name+'_'+expname+'_kappa_K='+str(K)+'.csv',params['kappa'])
    elif exp == 1 or exp == 3:
        for k in range(K):
            np.savetxt('experiments/outputs'+expname+'/'+name+'_'+expname+'_L_K='+str(K)+'_k'+str(k)+'.csv',params['Lambda'][k])
    
    # test
    
    if exp==0:
        model = Watson_EM(K=K,p=p,params=params)
        test_loglik = model.log_likelihood(X=data_test)
    elif exp==1:
        model = ACG_EM(K=K,p=p,params=params)
        test_loglik = model.log_likelihood(X=data_test)
    if exp==2:
        model = Watson_torch(K=K,p=p,params=params)
        with torch.no_grad():
            test_loglik = model.log_likelihood(X=torch.tensor(data_test))
    elif exp==3:
        model = ACG_torch(K=K,p=p,rank=p,params=params)
        with torch.no_grad():
            test_loglik = model.log_likelihood(X=torch.tensor(data_test))
    
    np.savetxt('experiments/outputs'+expname+'/'+name+'_'+expname+'_traintestlikelihood'+str(K)+'.csv',np.array([loglik[-1],test_loglik]))

    stop=7


if __name__=="__main__":
    # run_experiment(exp=int(1))
    for m in range(4):
        run_experiment(exp=int(m))
    # run_experiment(exp=int(sys.argv[1]))