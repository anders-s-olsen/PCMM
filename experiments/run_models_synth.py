import numpy as np
import torch
from src.helper_functions.helper_functions import load_data,train_model,test_model

torch.set_default_dtype(torch.float64)
import sys
import os

tol = 1e-10
num_repl_outer = 10
num_repl_inner = 1
num_iter = 100000

def run_experiment(modelname,LR,init):

    if LR==0:
        os.environ["OMP_NUM_THREADS"] = '8'
    else:
        os.environ["OMP_NUM_THREADS"] = '1'
        torch.set_num_threads(8)
        torch.set_num_interop_threads(8)

    ps = [3,10,25]
    Ks = [2,5,10]
    for p in ps:
        for K in Ks:
            allliks = np.zeros((2,num_repl_outer))
            if K>=p:
                continue
            if modelname=='Watson' or modelname=='ACG':
                data_train,data_test,_ = load_data(type='synth',num_subjects=None,num_eigs=1,LR=LR,p=p,K=K)
            elif modelname=='MACG':
                data_train,data_test,_ = load_data(type='synth',num_subjects=None,num_eigs=2,LR=LR,p=p,K=K)
    
            expname = '3d_'+init+'_'+str(LR)+'_p'+str(p)+'_K'+str(K)
            os.makedirs('experiments/synth_outputs',exist_ok=True)

            ### EM algorithms
            print('starting K='+str(K))
            rep_order = np.arange(num_repl_outer)
            np.random.shuffle(rep_order)
            for repl in range(num_repl_outer):
                rep = rep_order[repl]
                params,train_loglik,_ = train_model(modelname,K,data_train,p,init,LR,num_repl_inner,num_iter,tol)
                test_loglik,_ = test_model(modelname,K,data_test,params,LR,p)
                allliks[:,repl] = np.array([train_loglik,test_loglik])
                np.savetxt('experiments/synth_outputs/'+modelname+'_'+expname+'_traintestlikelihood.csv',allliks)

if __name__=="__main__":
    # run_experiment(modelname='ACG',LR=float(0),init='dc')
    # inits = ['unif','++','dc']
    # LRs = [0,0.01,0.1,1]
    # for init in inits:
    #     for LR in LRs:
    #         for m in range(2):
    #             run_experiment(mod=int(m),LR=LR,init=init)
    run_experiment(modelname=sys.argv[1],LR=float(sys.argv[2]),init=sys.argv[3])