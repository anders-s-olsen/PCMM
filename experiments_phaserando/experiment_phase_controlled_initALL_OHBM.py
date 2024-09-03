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
    df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
    return params,df

def run_experiment(extraoptions={},dataset='phase_controlled',suppress_output=False):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-10
    options['num_repl_outer'] = 25
    options['num_repl_inner'] = 1
    options['max_iter'] = 100000
    options['outfolder'] = 'data/results/torchvsEM_phase_controlled_results'
    options['num_subjects'] = 1
    options['data_type'] = dataset
    options['threads'] = 8
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    os.makedirs(options['outfolder'],exist_ok=True)
    p = 116
    K = 5
    ranks = [1,5,15]
    P = np.double(make_true_mat(options['num_subjects']))

    data_folder = 'data/synthetic/'
    options['experiment_name'] = dataset+'_'+options['modelname']+'_initALL'
    if options['modelname'] in ['Complex_Watson','Complex_ACG']:
        add_complex = 'complex_'
    else:
        add_complex = ''
    data_file = data_folder+add_complex+dataset+'_116data_eida.h5'
    with h5.File(data_file,'r') as f:
        data = f['U'][:]
        L = f['L'][:]

    if options['modelname'] in ['Watson','Complex_Watson','ACG','Complex_ACG']:
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
                        if options['modelname'] in ['Watson','Complex_Watson','ACG','Complex_ACG']:
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
                    if options['modelname'] in ['Watson','Complex_Watson']:        
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
        run_experiment(extraoptions=options,dataset=sys.argv[2],suppress_output=True)
    else:
        modelnames = ['Watson','Complex_Watson','ACG','Complex_ACG','MACG','SingularWishart']
        modelnames = ['Complex_ACG']
        for modelname in modelnames:
            run_experiment(extraoptions={'modelname':modelname},dataset='phase_narrowband_controlled',suppress_output=False)