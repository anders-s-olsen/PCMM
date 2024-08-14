from src.helper_functions import train_model,test_model
from src.load_HCP_data import load_real_data
import pandas as pd
import numpy as np
import torch
import os
import h5py as h5

def run(data_train,L_train,data_test,L_test,K,df,options,params=None,suppress_output=False,inner=None,p=116):
    params,train_posterior,loglik_curve = train_model(data_train,L_train,K=K,options=options,suppress_output=suppress_output,samples_per_sequence=1200,params=params)
    test_loglik,test_posterior = test_model(data_test=data_test,L_test=L_test,params=params,K=K,options=options)
    entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),'K':K,'p':p,'rank':options['rank'],'inner':inner,'iter':len(loglik_curve),'train_loglik':loglik_curve[-1],'test_loglik':test_loglik}
    df = pd.concat([df,pd.DataFrame([entry])],ignore_index=True)
    df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
    return params,df

def run_experiment(extraoptions={},suppress_output=False):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-8
    options['num_repl_outer'] = 10
    options['num_repl_inner'] = 1
    options['max_iter'] = 100000
    options['outfolder'] = 'data/results/116_results'
    options['num_subjects'] = 100
    options['data_type'] = 'real'
    options['threads'] = 8
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    os.makedirs(options['outfolder'],exist_ok=True)
    p = 116
    K = 5
    ranks = [1,5,15]
    options['experiment_name'] = 'realdata_'+options['modelname']

    # load data using h5
    with h5.File('data/synthetic/phase_controlled_116data_eida.h5','r') as f:
        data = f['U'][:]
        L = f['L'][:]
    if options['modelname']=='Watson' or options['modelname']=='ACG':
        data = data[:,:,0]
    data = data[:1200*options['num_subjects']]
    L = L[:1200*options['num_subjects']]
    df = pd.DataFrame()

    for inner in range(options['num_repl_outer']):        
        for LR in [0,0.1]:
            options['LR'] = LR

            if options['LR']!=0:
                data_train = torch.tensor(data)
                L_train = torch.tensor(L)
            else:
                data_train = data
                L_train = L

            for i,rank in enumerate(ranks):
                options['rank'] = rank

                if i==0: #for the first rank, initialize with dc_seg for EM
                    print('Running model:',options['modelname'],'rank:',rank,'LR:',LR,'inner:',inner)
                    options['HMM'] = False

                    if LR==0: #restart initialization
                        params_MM = None
                        if options['modelname'] in ['Watson','ACG']:
                            options['init'] = 'dc_seg' #rank 1 model
                        elif options['modelname']=='MACG':
                            options['init'] = 'gc_seg'
                        elif options['modelname']=='SingularWishart':
                            options['init'] = 'wgc_seg'

                    else: #load EM model
                        params_MM = np.load(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_params.npy',allow_pickle=True).item()
                        for key in params_MM.keys():
                            params_MM[key] = torch.tensor(params_MM[key])
                        options['init'] = 'no'

                    params,df = run(data_train=data_train,L_train=L_train,K=K,P=P,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)
                    if LR==0:
                        np.save(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_params.npy',params)

                    if options['LR'] != 0: #rank 1 HMM
                        options['init'] = 'no'
                        options['HMM'] = True
                        _,df = run(data_train=data_train,L_train=L_train,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)

                else:
                    if options['modelname'] == 'Watson':        
                        break
                    print('Running model:',options['modelname'],'rank:',rank,'LR:',LR,'inner:',inner)
                    options['init'] = 'no'
                    options['HMM'] = False

                    if LR == 0: #load previous rank
                        params_MM = np.load(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(ranks[i-1])+'_params.npy',allow_pickle=True).item()
                    else:
                        params_MM = np.load(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_params.npy',allow_pickle=True).item()
                        for key in params_MM.keys():
                            params_MM[key] = torch.tensor(params_MM[key])
                            
                    params,df = run(data_train=data_train,L_train=L_train,K=K,P=P,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)
                    if LR==0:
                        np.save(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_params.npy',params)

                    if options['LR'] != 0: #rank X HMM
                        options['HMM'] = True
                        _,df = run(data_train=data_train,L_train=L_train,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)


if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        print(sys.argv)
        options = {}
        options['modelname'] = sys.argv[1]
        run_experiment(extraoptions=options,suppress_output=True)
    else:
        modelnames = ['Watson','ACG','MACG','SingularWishart']
        modelnames = ['ACG']
        for modelname in modelnames:
            run_experiment(extraoptions={'modelname':modelname},suppress_output=False)