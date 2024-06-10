from src.helper_functions import train_model,calc_NMI,make_true_mat
import pandas as pd
import numpy as np
import torch
import os
import h5py as h5
# make an 2xN array where the first row is ones for the first half of N and zeros for the second half, opposite for the second row

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
    ranks = [1,5,10,25,50,'fullrank']

    # load data using h5
    with h5.File('data/synthetic/phase_controlled_116data_eida.h5','r') as f:
        data = f['U'][:]
    if options['modelname']=='Watson' or options['modelname']=='ACG' or options['modelname']=='ACG_lowrank':
        data = data[:,:,0]

    for inner in range(options['num_repl_outer']):        
        for LR in [0,0.01]:
            options['LR'] = LR
            if options['LR'] == 0:
                options['experiment_name'] = 'phase_controlled_EM_'+options['modelname']+'_initEM'
            else:
                options['experiment_name'] = 'phase_controlled_torch_'+options['modelname']+'_initEM'

            if inner==0:
                df = pd.DataFrame()
            else:
                df = pd.read_csv(options['outfolder']+'/'+options['experiment_name']+'.csv',index_col=0)

            if options['LR']!=0:
                data_train = torch.tensor(data)
            else:
                data_train = data

            for i,rank in enumerate(ranks):
                options['rank'] = rank

                if i==0: #for the first rank, initialize with dc_seg for EM
                    options['HMM'] = False
                    if LR==0: #restart initialization
                        params_MM = None
                        options['init'] = 'dc_seg' #rank 1 model
                    else: #load EM model
                        params_MM = np.load(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_params.npy',allow_pickle=True).item()
                        for key in params_MM.keys():
                            params_MM[key] = torch.tensor(params_MM[key])
                        options['init'] = 'no'
                    params,df = run(data_train=data_train,K=K,P=P,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)

                    if options['LR'] != 0: #rank 1 HMM
                        options['init'] = 'no'
                        options['HMM'] = True
                        _,df = run(data_train=data_train,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)

                elif rank=='fullrank':
                    if options['LR'] == 0 and options['modelname'] != 'Watson': #fullrank version only for EM and not watson
                        options['init'] = 'no'
                        options['HMM'] = False
                        params_MM = np.load(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(ranks[i-1])+'_params.npy',allow_pickle=True).item()
                        params_MM['Lambda'] = params_MM['M']@np.swapaxes(params_MM['M'],-2,-1)+np.eye(p)[None]
                        _,df = run(data_train=data_train,K=K,P=P,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)
                
                else:
                    if options['modelname'] == 'Watson':        
                        break
                    options['init'] = 'no'
                    options['HMM'] = False
                    if LR == 0: #load previous rank
                        params_MM = np.load(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(ranks[i-1])+'_params.npy',allow_pickle=True).item()
                    else:
                        params_MM = np.load(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_params.npy',allow_pickle=True).item()
                        for key in params_MM.keys():
                            params_MM[key] = torch.tensor(params_MM[key])
                    params,df = run(data_train=data_train,K=K,P=P,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)
                    if options['LR'] != 0: #rank X HMM
                        options['HMM'] = True
                        _,df = run(data_train=data_train,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)


if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        print(sys.argv)
        options = {}
        options['modelname'] = sys.argv[1]
        run_experiment(extraoptions=options,suppress_output=True)
    else:
        run_experiment(extraoptions={'modelname':'MACG'},suppress_output=False)
