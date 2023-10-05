import numpy as np
from src.models_python import MACGLowrankEM, ACGLowrankEM
from src.helper_functions import load_data

# options pertaining to current experiment
options = {}
options['tol'] = 1e-8
options['num_repl_outer'] = 10
options['num_subjects'] = 20
options['data_type'] = 'fMRI_SchaeferTian116_GSR'
options['LR'] = 0
K=2
options['init'] = 'uniform'

# we vary rank and n and conduct the same fixed point estimation using uniformly random initialization
# all for the p=116 GSR dataset. Then we evaluate norm between learned parameters, all for a single-component model


mean_norm = np.zeros((2,7,150))
for k,model in enumerate(['ACG','MACG']):
    options['modelname'] = model
    data_train,data_test,data_test2 = load_data(options=options)
    p = data_train.shape[1]
    for i,rank in enumerate([1,5,10,25,50,75,100]):
        ACG = ACGLowrankEM.ACG(K=K, p=p,rank=rank)
        MACG = MACGLowrankEM.MACG(K=K, p=p,q=2,rank=rank)
        options['ACG_rank'] = rank
        for j,n in enumerate(range(100,15001,100)):
            Z_all = np.zeros((10,p,p))
            for rep in range(10):
                if model=='ACG':
                    ACG.initialize(X=data_train,init='uniform',tol=options['tol'])
                    M = ACG.M_MLE_lowrank(M=ACG.M[0],X=data_train[:n],weights = None,tol=1e-10,max_iter=10000)
                elif model=='MACG':
                    MACG.initialize(X=data_train,init='uniform',tol=options['tol'])
                    M = MACG.M_MLE_lowrank(M=MACG.M[0],X=data_train[:n],weights = None,tol=1e-10,max_iter=10000)
                Z_all[rep] = M@M.T+np.eye(p)
            # compute the pairwise norm between Z=M@M.T+I for each rep
            norms = np.zeros(10)
            for rep in range(2,10):
                norms[rep] = np.linalg.norm(Z_all[rep]-Z_all[rep-1])
            norms[0] = np.linalg.norm(Z_all[0]-Z_all[-1])
            mean_norm[k,i,j] = np.mean(norms)
            print('model: '+model+', rank: '+str(rank)+', n: '+str(n)+', avg norm: '+str(np.mean(norms)))
        np.savetxt('data/results/ACG_MACG_fixedpointupdate_n_vs_p_'+model+'.csv',mean_norm[k],delimiter=',')


