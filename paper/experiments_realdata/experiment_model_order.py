from helper_functions_realdata import load_fMRI_data, run
import pandas as pd
import os
import numpy as np
import glob

def run_experiment(extraoptions={},suppress_output=False,GSR='GSR',df=None):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-10
    options['num_repl_outer'] = 1
    options['num_repl'] = 1
    options['max_iter'] = 100000
    options['outfolder'] = 'paper/data/results/116_results'
    options['HMM'] = False
    options['decrease_lr_on_plateau'] = False
    options['num_comparison'] = 50
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    os.makedirs(options['outfolder'],exist_ok=True)
    os.makedirs(options['outfolder']+'/params',exist_ok=True)

    # set parameters
    p = 116
    K = options['K']
    data_folder = 'paper/data/processed/'
    if GSR=='GSR':
        options['experiment_name'] = options['dataset']+'modelorder_realdata_'+options['modelname']+'_K='+str(K)+'_rank='+str(options['rank'])
        data_file = data_folder+options['dataset']+'fMRI_SchaeferTian116_GSR.h5'
    else:
        options['experiment_name'] = options['dataset']+'modelorder_realdata_'+options['modelname']+'_K='+str(K)+'_rank='+str(options['rank'])+'_noGSR'
        data_file = data_folder+options['dataset']+'fMRI_SchaeferTian116.h5'
    options['LR'] = 0.1

    if df is None:
        df = pd.DataFrame()

    if options['modelname'] in ['Normal','Complex_Normal','ACG','Complex_ACG','MACG','SingularWishart'] and options['rank']>1:
        # find the parameter file corresponding to a lower rank
        if GSR=='GSR':
            files_same_K = glob.glob(options['outfolder']+'/params/'+options['dataset']+'modelorder_realdata_'+options['modelname']+'_K='+str(K)+'_rank=*')
            files_same_K = [f for f in files_same_K if 'noGSR' not in f]
            ranks = np.array([int(f.split('rank=')[1].split('.')[0]) for f in files_same_K])
            lower_rank = ranks[ranks<options['rank']]
            if lower_rank.size!=0:
                params_MM = np.load(options['outfolder']+'/params/'+options['dataset']+'modelorder_realdata_'+options['modelname']+'_K='+str(K)+'_rank='+str(lower_rank.max())+'.npy',allow_pickle=True).item()
            else:
                raise ValueError('No lower rank model found')
        else:
            files_same_K = glob.glob(options['outfolder']+'/params/'+options['dataset']+'modelorder_realdata_'+options['modelname']+'_K='+str(K)+'_rank=*noGSR.npy')
            ranks = np.array([int(f.split('rank=')[1].split('_')[0]) for f in files_same_K])
            lower_rank = ranks[ranks<options['rank']]
            if lower_rank.size!=0:
                params_MM = np.load(options['outfolder']+'/params/'+options['dataset']+'modelorder_realdata_'+options['modelname']+'_K='+str(K)+'_rank='+str(lower_rank.max())+'_noGSR.npy',allow_pickle=True).item()
            else:
                raise ValueError('No lower rank model found')
    else:
        params_MM = None

    data_train,data_test1,data_test2 = load_fMRI_data(data_file,options,remove_first_ten=False)
    # data_train = data_train[:2*1940]
 
    print('Running model:',options['modelname'],'rank:',options['rank'],'inner:',options['inner'])

    if options['modelname'] in ['Watson','ACG','Complex_Watson','Complex_ACG']:
        options['init'] = 'dc_seg' #rank 1 model
    elif options['modelname']=='MACG':
        options['init'] = 'gc_seg'
    elif options['modelname']=='SingularWishart':
        options['init'] = 'wgc_seg'
    elif options['modelname'] in ['Normal','Complex_Normal']:
        options['init'] = 'ls_seg'
    else:
        options['init'] = '++'
    if options['modelname'] in ['Normal','Complex_Normal','Complex_ACG','ACG','MACG','SingularWishart'] and options['rank']>1:
        options['init'] = 'no'

    params,df,all_train_posterior,all_test_posterior = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=options['inner'],p=p)
    
    # options['HMM'] = True
    # options['init'] = 'no'
    # _,df,all_test_posterior_HMM = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,df=df,options=options,params=params,suppress_output=suppress_output,inner=options['inner'],p=p)

    np.savetxt(options['outfolder']+'/posteriors/'+options['experiment_name']+'_inner'+str(options['inner'])+'_train.txt',all_train_posterior,delimiter=',')
    np.savetxt(options['outfolder']+'/posteriors/'+options['experiment_name']+'_inner'+str(options['inner'])+'_test.txt',all_test_posterior,delimiter=',')

    if options['modelname'] in ['Normal','Complex_Normal','Complex_ACG','ACG','MACG','SingularWishart']:
        np.save(options['outfolder']+'/params/'+options['experiment_name']+'.npy',params)
    df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
    os.makedirs(options['outfolder']+'/posteriors',exist_ok=True)
    return df

if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        print(sys.argv)
        options = {}
        options['modelname'] = sys.argv[1]
        options['K'] = int(sys.argv[2])
        if options['modelname'] in ['Normal','Complex_Normal','Complex_ACG','ACG','MACG','SingularWishart']:
            options['dataset'] = sys.argv[3]
            GSR = sys.argv[4]
            df = None
            for inner in range(1):
                options['inner'] = inner
                for rank in [1,5,10,25,50,100]:#
                    options['rank'] = rank
                    df = run_experiment(extraoptions=options,suppress_output=True,GSR=GSR,df=df)
        else:
            options['rank'] = 25
            options['dataset'] = sys.argv[3]
            GSR = sys.argv[4]
            df = None
            for inner in range(1):
                options['inner'] = inner
                df = run_experiment(extraoptions=options,suppress_output=True,GSR=GSR,df=df)
    else:
        modelnames = ['Watson','ACG','MACG','SingularWishart','Complex_Watson','Complex_ACG','Normal','Complex_Normal','least_squares','diametrical','complex_diametrical','grassmann','weighted_grassmann']
        modelnames = ['Complex_ACG']
        # modelnames = ['weighted_grassmann','grassmann','complex_diametrical','diametrical','least_squares','Normal','Complex_Normal']
        K = 7
        dataset =  'all_tasks' # 'all_tasks', 'REST1REST2'
        ranks = [25] # [1,10,25,50,100]
        GSR = 'noGSR'
        # rank = 'fullrank'
        for inner in range(1):
            for rank in ranks:
                for modelname in modelnames:
                    run_experiment(extraoptions={'modelname':modelname,'K':K,'rank':rank,'dataset':dataset,'inner':inner},suppress_output=False,GSR=GSR)
                    