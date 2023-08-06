import numpy as np
import torch
from src.helper_functions.helper_functions import load_data,run_model,test_model

torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)
import sys
import os

tol = 1
num_iter = 100000
num_repl_outer = 10
num_repl_inner = 1
ranks = np.arange(1,454/2)

def run_experiment(modelname,LR,init,K):
    ## load data, only the first 200 subjects (each with 1200 data points)
    num_subjects = 200
    if modelname=='Watson' or modelname=='ACG':
        data_train,data_test = load_data(type='fMRI_SchaeferTian454',num_subjects=num_subjects,num_eigs=1,LR=LR)
    elif modelname=='MACG':
        data_train,data_test = load_data(type='fMRI_SchaeferTian454',num_subjects=num_subjects,num_eigs=2,LR=LR)
    
    os.makedirs('experiments/454_outputs',exist_ok=True)
    expname = '454_full_'+init+'_'+str(LR)+'_p'+str(data_train.shape[1])+'_K'+str(K)
    rep_order = np.arange(num_repl_outer)
    np.random.shuffle(rep_order)

    for repl in range(num_repl_outer):
        rep = rep_order[repl]
        print('starting K='+str(K)+' rep='+str(rep))

        if modelname=='Watson': #no rank stuff
            params,train_loglik = run_model(modelname,K,data_train,r,init,LR,num_repl_inner,num_iter,tol)
            test_loglik = test_model(modelname,K,data_test,params,LR,r)
            np.savetxt('experiments/454_outputs/'+modelname+'_'+expname+'_traintestlikelihood_r'+str(rep)+'.csv',np.array([train_loglik,test_loglik]))
        else:
            params = None
            for r in ranks:
                if r>1:
                    init = 'no'
                if r>1 or os.path.isfile('experiments/454_outputs/Watson_'+expname+'_traintestlikelihood_r'+str(rep)+'_rank'+str(r)+'.csv'):
                    continue
                params,train_loglik = run_model(modelname,K,data_train,r,init,LR,num_repl_inner,num_iter,tol)
                test_loglik = test_model(modelname,K,data_test,params,LR,r)
                np.savetxt('experiments/454_outputs/'+modelname+'_'+expname+'_traintestlikelihood_r'+str(rep)+'_rank'+str(r)+'.csv',np.array([train_loglik,test_loglik]))


if __name__=="__main__":
    # run_experiment(modelname='Watson',LR=float(0.1),init='++',K=30)
    # inits = ['unif','++','dc']
    # LRs = [0,0.01,0.1,1]
    # for init in inits:
    #     for LR in LRs:
    #         for m in range(2):
    #             run_experiment(mod=int(m),LR=LR,init=init)
    run_experiment(modelname=sys.argv[1],LR=float(sys.argv[2]),init=sys.argv[3],K=int(sys.argv[4]))