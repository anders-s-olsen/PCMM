from src.helper_functions import train_model,calc_NMI,make_true_mat
import pandas as pd
import numpy as np
import torch
import os
import h5py as h5

def run(data_train,L_train,K,P,df,options,params=None,suppress_output=False,inner=None,p=116):
    params,train_posterior,loglik_curve = train_model(data_train,L_train,K=K,options=options,suppress_output=suppress_output,samples_per_sequence=1200,params=params)
    train_NMI = calc_NMI(P,np.double(np.array(train_posterior)))
    entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),'K':K,'p':p,'rank':options['rank'],'inner':inner,'iter':len(loglik_curve),'train_loglik':loglik_curve[-1],'train_NMI':train_NMI}
    df = pd.concat([df,pd.DataFrame([entry])],ignore_index=True)
    return params,df

def run_experiment(extraoptions={},suppress_output=False):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-8
    options['num_repl_outer'] = 10
    options['num_repl_inner'] = 1
    options['max_iter'] = 100000
    options['outfolder'] = 'data/results/torchvsEM_phase_controlled_results'
    options['num_subjects'] = 1
    options['data_type'] = 'phase_controlled'
    options['threads'] = 8
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    os.makedirs(options['outfolder'],exist_ok=True)
    p = 116
    K = 5
    ranks = [1,5,25]
    options['experiment_name'] = 'phase_controlled_'+options['modelname']+'_initKmeans'
    P = np.double(make_true_mat(options['num_subjects']))

    # load data using h5
    with h5.File('data/synthetic/phase_controlled_116data_eida.h5','r') as f:
        data = f['U'][:]
        L = f['L'][:]
    if options['modelname']=='Watson' or options['modelname']=='ACG':
        data = data[:,:,0]
    data = data[:1200*options['num_subjects']]
    L = L[:1200*options['num_subjects']]

    for inner in range(options['num_repl_outer']):        

        if inner==0:
            df = pd.DataFrame()
        else:
            df = pd.read_csv(options['outfolder']+'/'+options['experiment_name']+'.csv',index_col=0)
        for LR in [0,0.01]:
            options['LR'] = LR

            if options['LR']!=0:
                data_train = torch.tensor(data)
                L_train = torch.tensor(L)
            else:
                data_train = data
                L_train = L

            for i,rank in enumerate(ranks):
                options['rank'] = rank
                if rank>1 and options['modelname']=='Watson':
                    break

                params_MM = None
                options['HMM'] = False
                if options['modelname'] in ['Watson','ACG']:
                    options['init'] = 'dc' #rank 1 model
                elif options['modelname']=='MACG':
                    options['init'] = 'gc'
                elif options['modelname']=='SingularWishart':
                    options['init'] = 'wgc'

                params,df = run(data_train=data_train,L_train=L_train,K=K,P=P,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)

                if options['LR'] != 0: #rank 1 HMM
                    options['HMM'] = True
                    _,df = run(data_train=data_train,L_train=L_train,K=K,P=P,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)

        df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')


if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        print(sys.argv)
        options = {}
        options['modelname'] = sys.argv[1]
        run_experiment(extraoptions=options,suppress_output=True)
    else:
        modelnames = ['Watson','ACG','MACG','SingularWishart']
        modelnames = ['SingularWishart']
        for modelname in modelnames:
            run_experiment(extraoptions={'modelname':modelname},suppress_output=False)