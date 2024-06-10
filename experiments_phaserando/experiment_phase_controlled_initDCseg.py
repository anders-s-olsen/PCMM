from src.helper_functions import train_model,calc_NMI,make_true_mat
import pandas as pd
import numpy as np
import torch
import os
import h5py as h5
# make an 2xN array where the first row is ones for the first half of N and zeros for the second half, opposite for the second row
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
    options['init'] = 'dc_seg'
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    os.makedirs(options['outfolder'],exist_ok=True)
    p = 116
    K = 5

    # load data using h5
    with h5.File('data/synthetic/phase_controlled_116data_eida.h5','r') as f:
        data_train = f['U'][:]
    if options['modelname']=='Watson' or options['modelname']=='ACG' or options['modelname']=='ACG_lowrank':
        data_train = data_train[:,:,0]

    if options['LR'] == 0:
        options['experiment_name'] = 'phase_controlled_EM_'+options['modelname']
    else:
        options['experiment_name'] = 'phase_controlled_torch_'+options['modelname']
    df = pd.DataFrame()

    P = np.double(make_true_mat())
    for HMM in [False,True]:
        if HMM:
            if options['LR']==0:
                continue
            options['HMM'] = True
        else:
            options['HMM'] = False
        for ACG_rank in [1,5,10,25,50,'fullrank']:#'full'
            options['rank'] = ACG_rank
            if ACG_rank != 1 and options['modelname'] == 'Watson':
                continue
            if ACG_rank == 'fullrank' and options['LR']!=0: #fullrank only for EM
                continue
            for inner in range(options['num_repl_outer']):
                print('Training model')
                if options['LR']!=0:
                    data_train = torch.tensor(data_train)
                params,train_posterior,loglik_curve = train_model(data_train=data_train,K=K,options=options,suppress_output=suppress_output,samples_per_sequence=1200)
                train_NMI = calc_NMI(P,np.double(np.array(train_posterior)))
                entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),'K':K,'p':p,
                        'ACG_rank':options['ACG_rank'],'inner':inner,'iter':len(loglik_curve),
                        'train_loglik':loglik_curve[-1],'train_NMI':train_NMI}
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
        run_experiment(extraoptions={'modelname':'MACG','LR':0},suppress_output=False)
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