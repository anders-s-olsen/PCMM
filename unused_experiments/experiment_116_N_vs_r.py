from src.helper_functions import train_model,test_model,calc_NMI
from src.load_HCP_data import load_real_data
import pandas as pd
import numpy as np
import torch
import os

def run_experiment(extraoptions={},suppress_output=False):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-8
    options['num_repl_outer'] = 10
    options['num_repl_inner'] = 1
    options['max_iter'] = 100000
    options['outfolder'] = 'data/results/torchvsEM_116_results_N_increasing2'
    options['num_subjects'] = None
    options['ACG_rank'] = 'full' #for ACG and MACG
    options['data_type'] = '116'
    options['threads'] = 8
    # options['init'] = 'unif'
    options['HMM'] = False
    options['num_subjects'] = 9
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    os.makedirs(options['outfolder'],exist_ok=True)

    if options['init'] in ['Grassmann','Grassmann_seg'] and options['modelname'] != 'MACG':
        return
    if options['LR'] == 0:
        options['experiment_name'] = 'Real116_EM_'+options['modelname']+'_'+options['init']
    else:
        options['experiment_name'] = 'Real116_torch_'+options['modelname']+'_'+options['init']
    try:
        df = pd.read_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
    except:
        df = pd.DataFrame()

    subjectlist = np.loadtxt('100unrelatedsubjectsIDs.txt',dtype=int)
    data_train, data_test = load_real_data(options,'fMRI_SchaeferTian116_GSR',subjectlist,suppress_output=suppress_output)

    print(options)
    Ns = np.array([100,1000,10000])
    Ks = [2,5,10]
    for ACG_rank in [5,10,20,50,100,116,'fullrank']:#'full'
        options['ACG_rank'] = ACG_rank
        if ACG_rank != 116 and options['modelname'] == 'Watson':
            continue
        if ACG_rank == 'fullrank' and options['LR']!=0: #fullrank only for EM
            continue
        for K in Ks:
            for N in Ns:
                train_posts = []
                test_posts = []
                train_lls = []
                test_lls = []
                rows_list = []
                iters = []
                # if df is not empty
                if not df.empty:
                    df_tmp = df[(df['K']==K) & (df['N']==N) & (df['ACG_rank']==ACG_rank)]
                    if len(df_tmp) == options['num_repl_outer']:
                        print('skipping K='+str(K)+' N='+str(N))
                        continue
                for inner in range(options['num_repl_outer']):
                    # choose random N points in data_train
                    # data_train2 = data_train[np.random.choice(data_train.shape[0],N,replace=False)]
                    # data_test2 = data_test[np.random.choice(data_test.shape[0],N,replace=False)]
                    data_train2 = data_train[:N]
                    data_test2 = data_test[:N]
                    if options['LR']!=0:
                        data_train2 = torch.tensor(data_train2)
                        data_test2 = torch.tensor(data_test2)
                    
                    print('starting K='+str(K)+' N='+str(N)+' inner='+str(inner))
                    try:
                        params,train_posterior,loglik_curve = train_model(data_train=data_train2,K=K,options=options,suppress_output=suppress_output)
                        test_loglik,test_posterior = test_model(data_test=data_test2,params=params,K=K,options=options)
                        train_posts.append(np.array(train_posterior))
                        test_posts.append(np.array(test_posterior))
                        train_lls.append(loglik_curve[-1])
                        test_lls.append(test_loglik.item())
                        iters.append(len(loglik_curve))
                    except:
                        train_posts.append(np.nan)
                        test_posts.append(np.nan)
                        train_lls.append(np.nan)
                        test_lls.append(np.nan)
                        iters.append(np.nan)
                    
                for inner in range(options['num_repl_outer']):
                        # make an entry in the dataframe for each comparison 1-2, 2-3, ..., 10-1
                    try:
                        if inner == options['num_repl_outer']-1:
                            entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),'K':K,'p':data_train.shape[1],'N':N,
                                    'ACG_rank':options['ACG_rank'],'inner':inner,'iter':iters[inner],
                                    'train_loglik':train_lls[inner],'test_loglik':test_lls[inner],
                                    'train_NMI':calc_NMI(train_posts[0],train_posts[-1]),'test_NMI':calc_NMI(test_posts[0],test_posts[-1])}
                        else:
                            entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),'K':K,'p':data_train.shape[1],'N':N,
                                    'ACG_rank':options['ACG_rank'],'inner':inner,'iter':iters[inner],
                                    'train_loglik':train_lls[inner],'test_loglik':test_lls[inner],
                                    'train_NMI':calc_NMI(train_posts[inner],train_posts[inner+1]),'test_NMI':calc_NMI(test_posts[inner],test_posts[inner+1])}
                    except:
                        entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),'K':K,'p':data_train.shape[1],'N':N,
                                'ACG_rank':options['ACG_rank'],'inner':inner,'iter':iters[inner],
                                'train_loglik':train_lls[inner],'test_loglik':test_lls[inner],
                                'train_NMI':np.nan,'test_NMI':np.nan}
                    rows_list.append(entry)
                df = pd.concat([df,pd.DataFrame(rows_list)],ignore_index=True)
                df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')

if __name__=="__main__":

    import sys
    if len(sys.argv)>1:
        print(sys.argv)
        options = {}
        options['modelname'] = sys.argv[1]
        options['LR'] = float(sys.argv[2])
        options['init'] = sys.argv[3]
        run_experiment(extraoptions=options,suppress_output=True)
    else:
        # run_experiment(extraoptions={'modelname':'Watson','LR':0,'init':'++_seg'},suppress_output=False)
        modelnames = ['Watson']
        LRs = [0.1]
        inits = ['unif','++','dc','++_seg','dc_seg','Grassmann','Grassmann_seg']
        options = {}
        for modelname in modelnames:
            for LR in LRs:
                for init in inits:
                    options['modelname'] = modelname
                    options['LR'] = LR
                    options['init'] = init
                    run_experiment(extraoptions=options)
    # print(options)