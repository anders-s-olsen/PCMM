from src.helper_functions import train_model,test_model,calc_NMI
from src.helper_functions2 import load_real_data
import pandas as pd
import numpy as np
# import torch
import os

def run_experiment(extraoptions={},suppress_output=False):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1
    options['num_repl_outer'] = 10
    options['num_repl_inner'] = 1
    options['max_iter'] = 100000
    options['outfolder'] = 'data/results/ACGrank_full'
    options['num_subjects'] = None
    options['ACG_rank'] = 'full' #for ACG and MACG
    options['data_type'] = '116'
    options['threads'] = 8
    options['init'] = 'unif'
    options['HMM'] = False
    options['num_subjects'] = 10
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    os.makedirs(options['outfolder'],exist_ok=True)

    if options['LR'] == 0:
        options['experiment_name'] = 'Realfull_EM_'+options['modelname']
    else:
        options['experiment_name'] = 'Realfull_torch_'+options['modelname']
    try:
        df = pd.read_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
    except:
        df = pd.DataFrame()

    subjectlist = np.loadtxt('100unrelatedsubjectsIDs.txt',dtype=int)
    data_train, data_test = load_real_data(options,'fMRI_full_GSR',subjectlist,suppress_output=suppress_output)
    print('Loaded data, shape: ',data_train.shape)

    print(options)
    Ks = [2,5,10]
    for K in Ks:
        for inner in range(options['num_repl_outer']):
            for ACG_rank in np.arange(10,200,10):#'full'
                options['ACG_rank'] = ACG_rank
                # if options['LR']!=0:
                #     data_train = torch.tensor(data_train)
                #     data_test = torch.tensor(data_test)
                    
                print('starting K='+str(K)+' inner='+str(inner))
                if ACG_rank == 10:
                    params,train_posterior,loglik_curve = train_model(data_train=data_train,K=K,options=options,suppress_output=suppress_output)
                else:
                    params,train_posterior,loglik_curve = train_model(data_train=data_train,K=K,options=options,suppress_output=suppress_output,params=params)
                test_loglik,test_posterior = test_model(data_test=data_test,params=params,K=K,options=options)

                entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),'K':K,'p':data_train.shape[1],
                        'ACG_rank':options['ACG_rank'],'inner':inner,'iter':len(loglik_curve),
                        'train_loglik':loglik_curve[-1],'test_loglik':test_loglik}
                np.savez(options['outfolder']+'/'+options['experiment_name']+'_K'+str(K)+'_inner'+str(inner)+'_ACGrank'+str(options['ACG_rank'])+'.npz',
                        train_posterior=train_posterior,loglik_curve=loglik_curve,test_posterior=test_posterior,params=params)

                df = pd.concat([df,pd.DataFrame([entry])],ignore_index=True)
                df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')

if __name__=="__main__":

    import sys
    if len(sys.argv)>1:
        print(sys.argv)
        options = {}
        options['modelname'] = sys.argv[1]
        options['LR'] = float(sys.argv[2])
        run_experiment(extraoptions=options,suppress_output=True)
    else:
        run_experiment(extraoptions={'modelname':'ACG','LR':0},suppress_output=False)
        # modelnames = ['Watson','ACG','MACG']
        # LRs = [0]
        # inits = ['unif','++','dc']
        # options = {}
        # for modelname in modelnames:
        #     for LR in LRs:
        #         for init in inits:
        #             options['modelname'] = modelname
        #             options['LR'] = LR
        #             options['init'] = init
        #             run_experiment(extraoptions=options)
        # options['modelname'] = 'ACG'
        # options['LR'] = 0
        # options['init'] = 'unif'
    # print(options)