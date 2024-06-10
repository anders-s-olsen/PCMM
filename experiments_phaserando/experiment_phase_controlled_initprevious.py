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
    p = 116
    K = 5

    # load data using h5
    with h5.File('data/synthetic/phase_controlled_116data_eida.h5','r') as f:
        data_train = f['U'][:]
    if options['modelname']=='Watson' or options['modelname']=='ACG' or options['modelname']=='ACG_lowrank':
        data_train = data_train[:,:,0]
    if options['LR']!=0:
        data_train = torch.tensor(data_train)

    if options['LR'] == 0:
        options['experiment_name'] = 'phase_controlled_EM_'+options['modelname']+'_initprevious'
    else:
        options['experiment_name'] = 'phase_controlled_torch_'+options['modelname']+'_initprevious'
    df = pd.DataFrame()
    P = np.double(make_true_mat())

    for inner in range(options['num_repl_outer']):
        options['init'] = 'dc_seg' #restart initialization  
        options['HMM'] = False
        options['rank'] = 1
        params_MM,df = run(data_train=data_train,K=K,P=P,df=df,options=options,params=None,suppress_output=suppress_output,inner=inner,p=p)
        df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
        if options['LR'] != 0:
            options['init'] = 'no'
            options['HMM'] = True
            params_HMM,df = run(data_train=data_train,K=K,P=P,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)
            df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')

        if options['modelname'] != 'Watson':
            for rank in [5,10,25,50]:
                options['init'] = 'no'
                options['HMM'] = False
                options['rank'] = rank
                params_MM,df = run(data_train=data_train,K=K,P=P,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)
                df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
                if options['LR'] != 0:
                    options['HMM'] = True
                    params_HMM,df = run(data_train=data_train,K=K,P=P,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)
                    df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
            if options['LR'] == 0:
                options['init'] = 'no'
                options['HMM'] = False
                options['rank'] = 'fullrank'
                params_MM['Lambda'] = params_MM['M']@np.swapaxes(params_MM['M'],-2,-1)+np.eye(p)[None]
                params_MM,df = run(data_train=data_train,K=K,P=P,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)
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
