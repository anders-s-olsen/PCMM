import numpy as np
import torch
from src.models_python.WatsonMixtureEM import Watson as Watson_EM
from src.models_python.mixture_EM_loop import mixture_EM_loop
from src.models_python.diametrical_clustering import diametrical_clustering, diametrical_clustering_plusplus
from src.models_pytorch.diametrical_clustering_torch import diametrical_clustering_torch, diametrical_clustering_plusplus_torch

def initialize_pi_mu_M(init,K,p,X=None,tol=1e-8,r=1,init_M=False):
    pi = np.repeat(1/K,repeats=K)
    if init is None or init=='uniform' or init=='unif':
        mu = np.random.uniform(size=(p,K))
        mu = mu/np.linalg.norm(mu,axis=0)
    elif init == '++' or init == 'plusplus' or init == 'diametrical_clustering_plusplus':
        if torch.is_tensor(X):
            mu = diametrical_clustering_plusplus_torch(X=X,K=K)
        else:
            mu = diametrical_clustering_plusplus(X=X,K=K)
    elif init == 'dc' or init == 'diametrical_clustering':
        if torch.is_tensor(X):
            mu = diametrical_clustering_torch(X=X,K=K,max_iter=100000,num_repl=5,init='++',tol=tol)
        else:
            mu = diametrical_clustering(X=X,K=K,max_iter=100000,num_repl=5,init='++',tol=tol)
    elif init == 'WMM' or init == 'Watson' or init == 'W' or init == 'watson':
        W = Watson_EM(K=K,p=p)
        params,_,_,_ = mixture_EM_loop(W,np.array(X),init='dc')
        mu = params['mu']
        pi = params['pi']
    elif init=='test':
        mu = np.array([[1,1],[0,1],[0,1]])
        mu = mu/np.linalg.norm(mu,axis=0)

    if init_M:
        M = np.random.uniform(size=(K,p,r))
        if init is not None and init !='unif' and init!='uniform':
            for k in range(K):
                M[k,:,0] = mu[:,k] #initialize only the first of the rank D columns this way, the rest uniform
        elif init == 'test':
            M1 = torch.tensor(np.loadtxt('data/test116M.txt'))
            M2 = torch.tensor(np.loadtxt('data/test116M2.txt'))
            M = np.stack([M1,M2],axis=0)
    else: M=None
    return pi, mu, M