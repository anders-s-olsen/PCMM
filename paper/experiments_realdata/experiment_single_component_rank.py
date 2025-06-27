from paper.helper_functions_paper import load_fMRI_data, run
import pandas as pd
import numpy as np
import os

def run_experiment(extraoptions={},suppress_output=False):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-10
    options['num_repl_outer'] = 1
    options['num_repl'] = 1
    options['max_iter'] = 100000
    options['outfolder'] = 'paper/data/results/116_results'
    options['HMM'] = False
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    os.makedirs(options['outfolder'],exist_ok=True)
    os.makedirs(options['outfolder']+'/params',exist_ok=True)
    os.makedirs(options['outfolder']+'/lls',exist_ok=True)

    # set parameters
    p = 116
    K = options['K']
    ranks = np.concatenate([np.ones(1).astype(int),np.arange(2,10,2).astype(int),np.arange(10,116,5).astype(int)])
    # ranks = np.arange(100,116,5).astype(int)
    options['experiment_name'] = options['dataset']+'rank_realdata_'+options['modelname']+'_K='+str(K)
    options['LR'] = 0.1

    df = pd.DataFrame()
    num_done = 0

    data_folder = 'paper/data/processed/'
    data_file = data_folder+options['dataset']+'fMRI_SchaeferTian116_GSR.h5'
    data_train,data_test1,data_test2 = load_fMRI_data(data_file,options)

    for inner in range(num_done,options['num_repl_outer']):   
        for i,rank in enumerate(ranks):
            options['rank'] = rank

            if i==0: #for the first rank, initialize with dc_seg for EM
                print('Running model:',options['modelname'],'rank:',rank,'inner:',inner)

                params_MM = None
                if options['modelname'] in ['Watson','ACG','Complex_Watson','Complex_ACG']:
                    options['init'] = 'dc' #rank 1 model
                elif options['modelname']=='MACG':
                    options['init'] = 'gc'
                elif options['modelname']=='SingularWishart':
                    options['init'] = 'wgc'
                elif options['modelname'] in ['Normal','Complex_Normal']:
                    options['init'] = 'ls_seg'
                else:
                    options['init'] = '++'

                params,df,_ = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)
                np.save(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_K'+str(K)+'_params.npy',params)

            else:
                if options['modelname'] in ['Watson','Complex_Watson']:        
                    break
                print('Running model:',options['modelname'],'rank:',rank,'inner:',inner)
                options['init'] = 'no'

                params_MM = np.load(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(ranks[i-1])+'_K'+str(K)+'_params.npy',allow_pickle=True).item()
                params,df,_ = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)
                np.save(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_K'+str(K)+'_params.npy',params)
            
            df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')

if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        print(sys.argv)
        options = {}
        options['modelname'] = sys.argv[1]
        options['K'] = int(sys.argv[2])
        options['dataset'] = sys.argv[3]
        run_experiment(extraoptions=options,suppress_output=True)
    else: #test
        modelnames = ['Complex_Normal']#'Complex_ACG',
        dataset = 'REST1REST2' # 'REST' or 'MOTOR' or 'SOCIAL' REST1REST2
        # modelnames = ['Complex_ACG']
        K = 1
        for modelname in modelnames:
            run_experiment(extraoptions={'modelname':modelname,'K':K,'dataset':dataset},suppress_output=False)