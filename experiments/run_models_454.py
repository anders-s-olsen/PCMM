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
# import matplotlib.pyplot as plt



def run_experiment(mod,LR,init):
    ### load data, only the first 200 subjects (each with 1200 data points) (not the same subjects in train/test)
    # data_train = np.array(h5py.File('data/processed/fMRI_atlas_RL2.h5', 'r')['Dataset'][:,:240000]).T
    # n,p = data_train.shape
    # print('Loaded training data')
    # data_test = np.array(h5py.File('data/processed/fMRI_atlas_RL1.h5', 'r')['Dataset'][:,:240000]).T
    # print('Loaded test data, beginning fit')

    expname = '3d_'+init+'_'+str(LR)
    os.makedirs('experiments/outputs'+expname,exist_ok=True)

    data_train = np.loadtxt('data/synthetic/synth_data_ACG.csv',delimiter=',')
    data_test = np.loadtxt('data/synthetic/synth_data_ACG2.csv',delimiter=',')
    p=3

    num_repl_outer = 10
    num_repl_inner = 1
    K = 2

    ### EM algorithms
    print('starting K='+str(K))
    for rep in range(num_repl_outer):
        if mod==0:
            if LR==0:
                model = Watson_EM(K=K,p=p)
            else:
                model = Watson_torch(K=K,p=p)
            name='Watson'
        elif mod==1:
            if LR==0:
                model = ACG_EM(K=K,p=p)
            else:
                model = ACG_torch(K=K,p=p,rank=p) #cholesky formulation when full rank
            name='ACG'

        if LR==0: #EM
            params,_,loglik,num_iter = mixture_EM_loop(model,data_train,tol=1e-6,max_iter=100000,
                                                       num_repl=num_repl_inner,init=init)
            pi = params['pi']
            if mod == 0:    
                mu = params['mu']
                kappa = params['kappa']
            elif mod == 1:
                Lambda = params['Lambda']
        else:
            params,_,loglik,num_iter = mixture_torch_loop(model,torch.tensor(data_train),tol=1e-6,max_iter=100000,
                                                          num_repl=num_repl_inner,init=init,LR=LR)
            Softmax = torch.nn.Softmax(dim=0)
            Softplus = torch.nn.Softplus(beta=20, threshold=1)
            pi = Softmax(params['pi'])
            if mod ==0:
                mu = torch.nn.functional.normalize(params['mu'],dim=0)
                kappa = Softplus(params['kappa'])
            elif mod == 1:
                Lambda = torch.zeros(K,p,p)
                for k in range(K):
                    L_tri_inv = params['L_tri_inv'][k]
                    Lambda[k] = torch.linalg.inv(L_tri_inv@L_tri_inv.T)
                    Lambda[k] = p*Lambda[k]/torch.trace(Lambda[k])

        # np.savetxt('experiments/outputs'+expname+'/'+name+'_'+expname+'_trainlikelihoodcurve_K='+str(K)+'_r'+str(rep)+'.csv',np.array(loglik))
        np.savetxt('experiments/outputs'+expname+'/'+name+'_'+expname+'_pi_K='+str(K)+'_r'+str(rep)+'.csv',pi)
        if mod==0:
            np.savetxt('experiments/outputs'+expname+'/'+name+'_'+expname+'_mu_K='+str(K)+'_r'+str(rep)+'.csv',mu)
            np.savetxt('experiments/outputs'+expname+'/'+name+'_'+expname+'_kappa_K='+str(K)+'_r'+str(rep)+'.csv',kappa)
        elif mod == 1:
            for k in range(K):
                np.savetxt('experiments/outputs'+expname+'/'+name+'_'+expname+'_L_K='+str(K)+'_k'+str(k)+'_r'+str(rep)+'.csv',Lambda[k])
        
        # test
        if mod==0:
            if LR==0:
                model = Watson_EM(K=K,p=p,params={'mu':mu,'kappa':kappa,'pi':pi})
                test_loglik = model.log_likelihood(X=data_test)
            else:
                model = Watson_torch(K=K,p=p,params={'mu':mu,'kappa':kappa,'pi':pi})
                with torch.no_grad():
                    test_loglik = model.test_log_likelihood(X=torch.tensor(data_test))
        elif mod==1:
            if LR==0:
                model = ACG_EM(K=K,p=p,params={'pi':pi,'Lambda':Lambda})
                test_loglik = model.log_likelihood(X=data_test)
            else:
                model = ACG_torch(K=K,p=p,rank=p,params={'pi':pi,'Lambda':Lambda}) #cholesky formulation when full rank
                with torch.no_grad():
                    test_loglik = model.test_log_likelihood(X=torch.tensor(data_test))
        
        np.savetxt('experiments/outputs'+expname+'/'+name+'_'+expname+'_traintestlikelihood'+str(K)+'_r'+str(rep)+'.csv',np.array([loglik[-1],test_loglik]))

    stop=7


if __name__=="__main__":
    # run_experiment(exp=int(2))
    # inits = ['unif','++','dc']
    # LRs = [0.01,0.1,1]
    # for init in inits:
    #     for LR in LRs:
    #         for m in range(4):
    #             run_experiment(exp=int(m),LR=LR,init=init)
    run_experiment(exp=int(sys.argv[1]),LR=float(sys.argv[2]),init=sys.argv[3])