import h5py
import numpy as np
import torch
from src.models_pytorch.ACG_lowrank_torch import ACG as ACG_torch
from src.models_pytorch.MACG_lowrank_torch import MACG as MACG_torch
from src.models_pytorch.Watson_torch import Watson as Watson_torch
from src.models_pytorch.mixture_torch_loop import mixture_torch_loop

torch.set_default_dtype(torch.float64)
import sys
import os


def run_experiment(mod,LR,init,K):
    ## load data, only the first 250 subjects (each with 1200 data points)
    data_train_tmp = np.array(h5py.File('data/processed/fMRI_full_RL2.h5', 'r')['Dataset'][:,:600000]).T
    if mod == 0 or mod == 1:
        data_train = torch.tensor(data_train_tmp[np.arange(600000,step=2),:])
    print('Loaded training data')
    data_test_tmp = np.array(h5py.File('data/processed/fMRI_full_RL1.h5', 'r')['Dataset'][:,:600000]).T
    if mod == 0 or mod == 1:
        data_test = torch.tensor(data_test_tmp[np.arange(600000,step=2),:])
    print('Loaded test data, beginning fit')
    p = data_train_tmp.shape[1]

    if mod == 2:
        data_train = np.zeros((300000,p,2))
        data_train[:,:,0] = data_train_tmp[np.arange(600000,step=2),:]
        data_train[:,:,1] = data_train_tmp[np.arange(600000,step=2)+1,:]
        data_train = torch.tensor(data_train)
        data_test = np.zeros((300000,p,2))
        data_test[:,:,0] = data_test_tmp[np.arange(600000,step=2),:]
        data_test[:,:,1] = data_test_tmp[np.arange(600000,step=2)+1,:]
        data_test = torch.tensor(data_test)

    # p=3
    # K=2
    # if mod == 0 or mod == 1:
    #     data_train = np.loadtxt('data/synthetic/synth_data_ACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
    #     data_test = np.loadtxt('data/synthetic/synth_data_ACG_p'+str(p)+'K'+str(K)+'_2.csv',delimiter=',')
    # elif mod==2:
    #     data_train = np.zeros((1000,p,2))
    #     data_train_tmp = np.loadtxt('data/synthetic/synth_data_MACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
    #     data_train[:,:,0] = data_train_tmp[np.arange(2000,step=2),:]
    #     data_train[:,:,1] = data_train_tmp[np.arange(2000,step=2)+1,:]
    #     data_test = np.zeros((1000,p,2))
    #     data_test_tmp = np.loadtxt('data/synthetic/synth_data_MACG_p'+str(p)+'K'+str(K)+'_2.csv',delimiter=',')
    #     data_test[:,:,0] = data_test_tmp[np.arange(2000,step=2),:]
    #     data_test[:,:,1] = data_test_tmp[np.arange(2000,step=2)+1,:]
    # data_train = torch.tensor(data_train)
    # data_test = torch.tensor(data_test)

    
    tol = 1
    num_iter = 100000

    num_repl_outer = 10
    num_repl_inner = 1
    Softmax = torch.nn.Softmax(dim=0)

    os.makedirs('experiments/454_outputs',exist_ok=True)

    if LR==0:
        os.environ["OMP_NUM_THREADS"] = '8'
    else:
        os.environ["OMP_NUM_THREADS"] = '1'
        torch.set_num_threads(8)
        torch.set_num_interop_threads(8)

    ranks = np.arange(1,454)
    
    expname = '454_full_'+init+'_'+str(LR)+'_p'+str(p)+'_K'+str(K)
    
    ### EM algorithms
    rep_order = np.arange(num_repl_outer)
    np.random.shuffle(rep_order)
    for repl in range(num_repl_outer):
        rep = rep_order[repl]
        print('starting K='+str(K)+' rep='+str(rep))
        for r in ranks:
            if mod == 0:
                if r>1 or os.path.isfile('experiments/454_outputs/'+name+'_'+expname+'_traintestlikelihood_r'+str(rep)+'_rank'+str(r)+'.csv'):
                    continue
            if mod==0:
                model = Watson_torch(K=K,p=p)
                name='Watson'
            elif mod==1:
                if r==1:
                    model = ACG_torch(K=K,p=p,rank=r) 
                else:
                    model = ACG_torch(K=K,p=p,rank=r,params=params)
                name='ACG'
            elif mod==2:
                if r==1:
                    model = MACG_torch(K=K,p=p,q=2,rank=r) 
                else:
                    model = MACG_torch(K=K,p=p,q=2,rank=r,params=params)
                name = 'MACG'
            
            if r == 1:
                params,_,loglik,_ = mixture_torch_loop(model,data_train,tol=tol,max_iter=100000,
                                                    num_repl=num_repl_inner,init=init,LR=LR)
            else:
                params,_,loglik,_ = mixture_torch_loop(model,data_train,tol=tol,max_iter=100000,
                                                    num_repl=num_repl_inner,init='no',LR=LR)                    
        
            pi = Softmax(params['pi'])
            if mod ==0:
                mu = torch.nn.functional.normalize(params['mu'],dim=0)
                # kappa = Softplus(params['kappa'])
                kappa = params['kappa']
            else:
                M = params['M']

            # np.savetxt('experiments/454_outputs/'+name+'_'+expname+'_trainlikelihoodcurve_r'+str(rep)+'.csv',np.array(loglik))
            # np.savetxt('experiments/454_outputs/'+name+'_'+expname+'_pi_r'+str(rep)+'.csv',pi)
            # if mod==0:
            #     np.savetxt('experiments/454_outputs/'+name+'_'+expname+'_mu_r'+str(rep)+'.csv',mu)
            #     np.savetxt('experiments/454_outputs/'+name+'_'+expname+'_kappa_r'+str(rep)+'.csv',kappa)
            # elif mod == 1:
            #     for k in range(K):
            #         np.savetxt('experiments/454_outputs/'+name+'_'+expname+'_L_k'+str(k)+'_r'+str(rep)+'.csv',Lambda[k])
            
            # test
            if mod==0:
                model = Watson_torch(K=K,p=p,params={'mu':mu,'kappa':kappa,'pi':pi})
                with torch.no_grad():
                    test_loglik = model.test_log_likelihood(X=data_test)
            elif mod==1:
                model = ACG_torch(K=K,p=p,rank=r,params={'pi':pi,'M':M}) #cholesky formulation when full rank
                with torch.no_grad():
                    test_loglik = model.test_log_likelihood(X=data_test)
            elif mod==2:
                model = MACG_torch(K=K,p=p,q=2,rank=r,params={'pi':pi,'M':M}) #cholesky formulation when full rank
                with torch.no_grad():
                    test_loglik = model.test_log_likelihood(X=data_test)                        
            
            np.savetxt('experiments/454_outputs/'+name+'_'+expname+'_traintestlikelihood_r'+str(rep)+'_rank'+str(r)+'.csv',np.array([loglik[-1],test_loglik]))

    stop=7


if __name__=="__main__":
    # run_experiment(mod=int(2),LR=float(0.1),init='++',K=30)
    # inits = ['unif','++','dc']
    # LRs = [0,0.01,0.1,1]
    # for init in inits:
    #     for LR in LRs:
    #         for m in range(2):
    #             run_experiment(mod=int(m),LR=LR,init=init)
    run_experiment(mod=int(sys.argv[1]),LR=float(sys.argv[2]),init=sys.argv[3],K=int(sys.argv[4]))