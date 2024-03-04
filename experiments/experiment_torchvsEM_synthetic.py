from src.helper_functions import load_synthetic_data,train_model,test_model,calc_NMI
import pandas as pd
import numpy as np
import torch
import os
# make an 2xN array where the first row is ones for the first half of N and zeros for the second half, opposite for the second row
def make_true_mat(K,N=10000):
    rows = []
    for k in range(K):
        row = np.zeros(N,dtype=bool)
        row[N//K*k:N//K*(k+1)] = 1
        rows.append(row)
    return np.vstack(rows)
def run_experiment(extraoptions={},suppress_output=False):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-8
    options['num_repl_outer'] = 10
    options['num_repl_inner'] = 1
    options['max_iter'] = 100000
    options['outfolder'] = 'data/results/torchvsEM_synthetic_results'
    options['num_subjects'] = None
    options['ACG_rank'] = 'full' #for ACG and MACG
    options['data_type'] = 'synthetic'
    options['threads'] = 8
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    os.makedirs(options['outfolder'],exist_ok=True)

    if options['LR'] == 0:
        options['experiment_name'] = 'Synthetic_EM_'+options['modelname']+'_'+options['init']
    else:
        options['experiment_name'] = 'Synthetic_torch_'+options['modelname']+'_'+options['init']
    try:
        df = pd.read_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
        # if HMM not a column, add it where its false everywhere
        if 'HMM' not in df.columns:
            df['HMM'] = 'False'
    except:
        df = pd.DataFrame()

    print(options)
    ps = [3,10,25]
    Ks = [2,5,10]
    N = 10000
    for HMM in [False,True]:
        if HMM:
            if options['LR']==0:
                continue
            options['HMM'] = True
        else:
            options['HMM'] = False
        for ACG_rank in ['lowrank','fullrank']:#'full'
            options['ACG_rank'] = ACG_rank
            if ACG_rank == 'fullrank' and options['modelname'] == 'Watson':
                continue
            if ACG_rank == 'fullrank' and options['LR']!=0: #fullrank only for EM
                continue
            for inner in range(options['num_repl_outer']):
                # if dataframe not empty
                # if not df.empty:
                #     dftmp = df[df['ACG_rank']==ACG_rank]
                #     dftmp2 = df[df['ACG_rank']!=ACG_rank]
                #     if len(dftmp[dftmp['inner']==inner])==6:
                #         continue
                #     elif len(dftmp[dftmp['inner']==inner])>0:
                #         dftmp = dftmp[dftmp['inner']!=inner]
                #         df = pd.concat([dftmp,dftmp2],ignore_index=True)
                rows_list = []
                for p in ps:
                    for K in Ks:
                        if K>=p:
                            continue
                        P = make_true_mat(K,N)
                        print('starting K='+str(K)+' p='+str(p)+' inner='+str(inner))
                        data_train,data_test = load_synthetic_data(options=options,p=p,K=K)
                        if options['LR']!=0:
                            data_train = torch.tensor(data_train)
                            data_test = torch.tensor(data_test)
                        params,train_posterior,loglik_curve = train_model(data_train=data_train,K=K,options=options,suppress_output=suppress_output)
                        test_loglik,test_posterior = test_model(data_test=data_test,params=params,K=K,options=options)
                        train_NMI = calc_NMI(P,np.array(train_posterior))
                        test_NMI = calc_NMI(P,np.array(test_posterior))
                        entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),'K':K,'p':p,
                                'ACG_rank':options['ACG_rank'],'inner':inner,'iter':len(loglik_curve),
                                'train_loglik':loglik_curve[-1],'test_loglik':test_loglik.item(),
                                'train_NMI':train_NMI,'test_NMI':test_NMI}
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
        run_experiment(extraoptions={'modelname':'Watson','LR':0.1,'init':'unif'},suppress_output=False)
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