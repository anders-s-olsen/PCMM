from src.helper_functions import train_model,calc_NMI,make_true_mat
import pandas as pd
import numpy as np
import torch
import os
import h5py as h5

def run(data_train,K,P,df,options,params=None,suppress_output=False,inner=None,p=116):
    params,train_posterior,loglik_curve = train_model(data_train,K=K,options=options,suppress_output=suppress_output,samples_per_sequence=1200,params=params)
    train_NMI = calc_NMI(P,np.double(np.array(train_posterior)))
    entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),'K':K,'p':p,'ACG_rank':options['ACG_rank'],'inner':inner,'iter':len(loglik_curve),'train_loglik':loglik_curve[-1],'train_NMI':train_NMI}
    df = pd.concat([df,pd.DataFrame([entry])],ignore_index=True)
    df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
    if options['LR']==0:
        np.save(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(options['ACG_rank'])+'_params.npy',params)
    return params,df

def run_experiment(extraoptions={},suppress_output=False):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-8
    options['num_repl_outer'] = 50
    options['num_repl_inner'] = 1
    options['max_iter'] = 100000
    options['outfolder'] = 'data/results/torchvsEM_phase_controlled_results'
    options['num_subjects'] = 10
    options['data_type'] = 'phase_controlled'
    options['threads'] = 8
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    os.makedirs(options['outfolder'],exist_ok=True)
    os.makedirs(options['outfolder']+'/params',exist_ok=True)
    p = 116
    K = 5
    P = np.double(make_true_mat())
    ranks = [1,5,10,25,50,116]
    options['LR'] = 1
    # load data using h5
    with h5.File('data/synthetic/phase_controlled_116data.h5','r') as f:
        data = torch.tensor(f['X'][:])

    for inner in range(options['num_repl_outer']):        
        options['experiment_name'] = 'phase_controlled_'+options['modelname']+'_unif'
        
        if inner==0:
            df = pd.DataFrame()
        else:
            df = pd.read_csv(options['outfolder']+'/'+options['experiment_name']+'.csv',index_col=0)
        for i,rank in enumerate(ranks):
            if options['modelname']!='MVG_lowrank' and rank>1:
                break

            options['rank'] = rank
            if i==0:
                options['HMM'] = False
                options['init'] = 'euclidean_kmeans_seg'
                params,df = run(data_train=data,K=K,P=P,df=df,options=options,params=None,suppress_output=suppress_output,inner=inner,p=p)

                options['HMM'] = True
                options['init'] = 'no'
                _,df = run(data_train=data,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)
            else:
                options['HMM'] = False
                options['init'] = 'no'
                params,df = run(data_train=data,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)

                options['HMM'] = True
                options['init'] = 'no'
                _,df = run(data_train=data,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)   


if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        print(sys.argv)
        options = {}
        options['modelname'] = sys.argv[1]
        run_experiment(extraoptions=options,suppress_output=True)
    else:
        run_experiment(extraoptions={'modelname':'MVG_scalar'},suppress_output=False)
