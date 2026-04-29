from paper.helper_functions_paper import load_fMRI_data, run
import pandas as pd
import os
import numpy as np
import glob

def run_experiment(extraoptions={},suppress_output=False,dataset='SchaeferTian116_GSR',df=None, train_or_not=True):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-7
    options['max_iter'] = 100000
    options['outfolder'] = 'paper/data/results/'+dataset
    options['HMM'] = False
    options['decrease_lr_on_plateau'] = False
    options['num_comparison'] = 50
    options['num_repl'] = 1
    options['first_N_poststim_volumes'] = 'all' #overwritten below if provided in extraoptions
    options.update(extraoptions) #modelname, LR, init controlled in shell script

    # set parameters
    p = 116
    K = options['K']
    data_folder = 'paper/data/processed/concatenated_datasets/'
    options['experiment_name'] = options['experiment']+'modelorder_realdata_'+options['modelname']+'_K='+str(K)+'_rank='+str(options['rank'])+'_inner='+str(options['inner'])
    
    if options.get('split') is not None:
        # options['experiment_name'] += '_split='+str(options['split'])
        data_file = data_folder+options['experiment']+dataset+'_8020split'+str(options['split'])+'.h5'
        options['outfolder'] += '_8020split'+str(options['split'])
    else:
        data_file = data_folder+options['experiment']+dataset+'.h5'
    options['LR'] = 0.1
    os.makedirs(options['outfolder'],exist_ok=True)
    os.makedirs(options['outfolder']+'/params',exist_ok=True)
    os.makedirs(options['outfolder']+'/posteriors',exist_ok=True)
    os.makedirs(options['outfolder']+'/dfs',exist_ok=True)

    if df is None:
        df = pd.DataFrame()

    if options['modelname'] in ['Normal','Complex_Normal','ACG','Complex_ACG','MACG','SingularWishart'] and options['rank']>1:
        files_same_K = glob.glob(options['outfolder']+'/params/'+options['experiment_name'].replace('_rank='+str(options['rank']),'_rank=*')+'.npy')
        # files_same_K = [f for f in files_same_K if '_GSR' not in f]
        ranks = np.array([int(f.split('rank=')[1].split('_')[0]) for f in files_same_K])
        # if train_or_not:
        lower_rank = ranks[ranks<options['rank']]
        # else:
        # lower_rank = ranks[ranks==options['rank']]
        if lower_rank.size!=0:
            params_MM = np.load(options['outfolder']+'/params/'+options['experiment_name'].replace('_rank='+str(options['rank']),'_rank='+str(lower_rank.max()))+'.npy',allow_pickle=True).item()
        else:
            raise ValueError('No lower rank model found')
    else:
        params_MM = None
    
    true_labels_train = None
    true_labels_test = None
    pts_per_subject_train = None
    pts_per_subject_test = None
    if options['first_N_poststim_volumes']=='cov': #covariance
        data_train,data_test1,data_test2 = load_fMRI_data(data_file,options, standardize=False, covariance=True)
    elif options['first_N_poststim_volumes']=='all': #time-series
        data_train,data_test1,data_test2 = load_fMRI_data(data_file,options, standardize=False, covariance=False)
    else: #only the first N post-stim volumes
        data_train,data_test1,data_test2,true_labels_train,true_labels_test, pts_per_subject_train, pts_per_subject_test = load_fMRI_data(data_file,options, standardize=False, covariance=False)
    
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

    params,df,all_train_posterior,all_test_posterior = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=options['inner'],p=p, true_labels_train=true_labels_train,true_labels_test=true_labels_test, pts_per_subject_train=pts_per_subject_train, pts_per_subject_test=pts_per_subject_test,train_or_not=train_or_not)
    
    np.savetxt(options['outfolder']+'/posteriors/'+options['experiment_name']+'_train.txt',all_train_posterior,delimiter=',')
    np.savetxt(options['outfolder']+'/posteriors/'+options['experiment_name']+'_test.txt',all_test_posterior,delimiter=',')

    if options['modelname'] in ['Normal','Complex_Normal','Complex_ACG','ACG','MACG','SingularWishart']:
        np.save(options['outfolder']+'/params/'+options['experiment_name']+'.npy',params)
    df.to_csv(options['outfolder']+'/dfs/'+options['experiment_name']+'.csv')
    return df
    
def run_func(args, options={}):
    split_idx = options.get('split', None)
    if split_idx is not None:
        num_inner = 5
    else:
        num_inner = 10
    print(args)
    # options = {}
    options['modelname'] = args[1]
    options['K'] = int(args[2])
    if options['modelname'] in ['Normal','Complex_Normal','Complex_ACG','ACG','MACG','SingularWishart']:
        options['experiment'] = args[3]
        dataset = args[4]
        for inner in range(num_inner):
            options['inner'] = inner
            for rank in [1,5,10,25,50,100]:#
                options['rank'] = rank
                df = run_experiment(extraoptions=options,suppress_output=True,dataset=dataset,df=None)
    else:
        options['rank'] = 25
        options['experiment'] = args[3]
        dataset = args[4]
        for inner in range(num_inner):
            options['inner'] = inner
            df = run_experiment(extraoptions=options,suppress_output=True,dataset=dataset,df=None)

if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        if int(sys.argv[-1])>0:
            extraoptions = {'split': sys.argv[-1]}
            # for split in range(5):
            #     extraoptions = {'split': split}
            run_func(sys.argv, options=extraoptions)
        else:
            run_func(sys.argv)
    else: # test
        modelnames = ['Watson','ACG','MACG','SingularWishart','Complex_Watson','Complex_ACG','Normal','Complex_Normal','least_squares','diametrical','complex_diametrical','grassmann','weighted_grassmann']
        modelnames = ['Complex_Normal'] #   'logistic','rbf-svm','linear-svm'
        # modelnames = ['weighted_grassmann','grassmann','complex_diametrical','diametrical','least_squares','Normal','Complex_Normal']
        K = 7
        ranks = [1] # [1,5,10,25,50,100]
        # train_or_not = False
        experiment =  'all_tasks' # 'all_tasks', 'REST1REST2'
        # inner = 6
        # for post_stim_vols in [10,15,20,'all']:#1,2,3,4,5,6,7,8,9,10,15,20,'cov','all'
        #     run_experiment(extraoptions={'modelname':modelnames[0],'K':K,'rank':ranks[0],'experiment':experiment,'inner':inner,'first_N_poststim_volumes':post_stim_vols},suppress_output=False,dataset='fMRI_SchaeferTian116_GSR',train_or_not=train_or_not)
        
         
        dataset = 'fMRI_SchaeferTian116_GSR' # 'fMRI_SchaeferTian232_GSR'
        # rank = 'fullrank'
        for inner in range(1):
            for rank in ranks:
                for modelname in modelnames:
                    run_experiment(extraoptions={'modelname':modelname,'K':K,'rank':rank,'experiment':experiment,'inner':inner,'split':4},suppress_output=False,dataset=dataset)
                    