import numpy as np
import torch
from src.helper_functions.helper_functions import load_data,train_model,test_model

torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)
import sys
import os

tol = 0.1
num_iter = 1000000
num_repl_outer = 10
num_repl_inner = 1
ranks = np.arange(start=1,stop=116,step=1)

def run_experiment(modelname,LR,init0,GSR):
    ## load data, only the first 100 subjects (each with 1200 data points)
    num_subjects = 50
    if GSR==0:
        type = 'fMRI_SchaeferTian116'
        outfolder = 'experiments/116_outputs'
    elif GSR==1:
        type = 'fMRI_SchaeferTian116_GSR'
        outfolder = 'experiments/116GSR_outputs'
    if modelname=='Watson' or modelname=='ACG':
        data_train,data_test,data_test2 = load_data(type=type,num_subjects=num_subjects,num_eigs=1,LR=LR)
    elif modelname=='MACG':
        data_train,data_test,data_test2 = load_data(type=type,num_subjects=num_subjects,num_eigs=2,LR=LR)
        data_train = data_train.swapaxes(-2,-1)
        data_test = data_test.swapaxes(-2,-1)
        data_test2 = data_test2.swapaxes(-2,-1)
    
    os.makedirs(outfolder,exist_ok=True)
    
    for K in np.arange(2,31):
        expname = '116_'+init0+'_'+str(LR)+'_p'+str(data_train.shape[1])+'_K'+str(K)
        rep_order = np.arange(num_repl_outer)
        np.random.shuffle(rep_order)

        for repl in range(num_repl_outer):
            rep = rep_order[repl]
            print('starting K='+str(K)+' rep='+str(rep))

            if modelname=='Watson': #no rank stuff
                if os.path.isfile(outfolder+'/Watson_'+expname+'_traintestlikelihood_r'+str(rep)+'.csv'):
                    continue
                params,train_loglik = train_model(modelname=modelname,K=K,data_train=data_train,rank=None,init=init0,LR=LR,num_repl_inner=num_repl_inner,num_iter=num_iter,tol=tol)
                test_loglik,_ = test_model(modelname=modelname,K=K,data_test=data_test,params=params,LR=LR,rank=None)
                test_loglik2,_ = test_model(modelname=modelname,K=K,data_test=data_test2,params=params,LR=LR,rank=None)
                np.savetxt(outfolder+'/'+modelname+'_'+expname+'_traintestlikelihood_r'+str(rep)+'.csv',np.array([train_loglik,test_loglik,test_loglik2]))
            else:
                params = None
                logliks = np.zeros((3,116))
                for r,rank in enumerate(ranks):
                    if rank>1:
                        init = 'no'
                    else:
                        init = init0
                    if rank==116:
                        params,train_loglik = train_model(modelname=modelname,K=K,data_train=data_train,rank=116,init=init0,LR=LR,num_repl_inner=num_repl_inner,num_iter=num_iter,tol=tol,params=None)
                    else:
                        params,train_loglik = train_model(modelname=modelname,K=K,data_train=data_train,rank=rank,init=init,LR=LR,num_repl_inner=num_repl_inner,num_iter=num_iter,tol=tol,params=params)
                    
                    test_loglik,_ = test_model(modelname=modelname,K=K,data_test=data_test,params=params,LR=LR,rank=rank)
                    test_loglik2,_ = test_model(modelname=modelname,K=K,data_test=data_test2,params=params,LR=LR,rank=rank)
                    logliks[0,r] = train_loglik
                    logliks[1,r] = test_loglik
                    logliks[2,r] = test_loglik2
                    np.savetxt(outfolder+'/'+modelname+'_'+expname+'_traintestlikelihood_r'+str(rep)+'.csv',logliks)


if __name__=="__main__":
    # run_experiment(modelname='MACG',LR=float(0.1),init0='unif',GSR=1)
    # inits = ['unif','++','dc']
    # LRs = [0,0.01,0.1,1]
    # for init in inits:
    #     for LR in LRs:
    #         for m in range(2):
    #             run_experiment(mod=int(m),LR=LR,init=init)
    run_experiment(modelname=sys.argv[1],LR=float(sys.argv[2]),init0=sys.argv[3],GSR=int(sys.argv[4]))