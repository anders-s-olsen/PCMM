import h5py
import numpy as np
from src.models_python import mixture_EM_loop, WatsonMixtureEM

# datah5_train = h5py.File('../data/processed/fMRI_atlas_RL1.h5', 'r')
data_train = np.array(h5py.File('../data/processed/fMRI_atlas_RL1.h5', 'r')['Dataset']).T
data_test = np.array(h5py.File('../data/processed/fMRI_atlas_RL2.h5', 'r')['Dataset']).T

for K in range(2,21):
    W = WatsonMixtureEM.Watson(K=K,p=data_train.shape[1])

    # train
    params_W,beta_W,loglik_W,num_iter = mixture_EM_loop.mixture_EM_loop(W,data_train,tol=1e-8,max_iter=10000,num_repl=1,init='diametrical_clustering')
    np.savetxt('outputs/Watson_454_trainlikelihoodcurve_K='+str(K)+'.csv',np.array(loglik_W))
    # test
    W = WatsonMixtureEM.Watson(K=K,p=data_train.shape[1],params=params_W)
    test_loglik = W.log_likelihood(X=data_test)
    np.savetxt('outputs/Watson_454_traintestlikelihood',+str(K)+'.csv',np.array([loglik_W[-1],test_loglik]))