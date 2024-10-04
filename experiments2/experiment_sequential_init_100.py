from src.helper_functions import train_model,test_model
import pandas as pd
import numpy as np
import os
import h5py as h5

def run(data_train,L_train,data_test,L_test,K,df,options,params=None,suppress_output=False,inner=None,p=116):
    params,train_posterior,loglik_curve = train_model(data_train,L_train,K=K,options=options,suppress_output=suppress_output,samples_per_sequence=1200,params=params)
    test_loglik,test_posterior = test_model(data_test=data_test,L_test=L_test,params=params,K=K,options=options)

    test_lls = np.zeros(100)
    train_lls = np.zeros(100)
    for sub in range(100):
        test_data = data_test[sub*1200:(sub+1)*1200]
        test_L = L_test[sub*1200:(sub+1)*1200]
        train_data = data_train[sub*1200:(sub+1)*1200]
        train_L = L_train[sub*1200:(sub+1)*1200]
        test_lls[sub],_ = test_model(data_test=test_data,L_test=test_L,params=params,K=K,options=options)
        train_lls[sub],_ = test_model(data_test=train_data,L_test=train_L,params=params,K=K,options=options)
    np.savetxt(options['outfolder']+'/lls/'+options['experiment_name']+'_train_ll_rank'+str(options['rank'])+'_inner'+str(inner)+'.txt',train_lls)
    np.savetxt(options['outfolder']+'/lls/'+options['experiment_name']+'_test_ll_rank'+str(options['rank'])+'_inner'+str(inner)+'.txt',test_lls)

    entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),
             'K':K,'p':p,'rank':options['rank'],'inner':inner,'iter':len(loglik_curve),
             'train_loglik':loglik_curve[-1],'test_loglik':test_loglik}
    #save the posterior as txt
    if K!=1:
        np.savetxt(options['outfolder']+'/posteriors/'+options['experiment_name']+'_train_posterior_rank'+str(options['rank'])+'_inner'+str(inner)+'.txt',train_posterior)
        np.savetxt(options['outfolder']+'/posteriors/'+options['experiment_name']+'_test_posterior_rank'+str(options['rank'])+'_inner'+str(inner)+'.txt',test_posterior)
    df = pd.concat([df,pd.DataFrame([entry])],ignore_index=True)
    df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
    return params,df

def run_experiment(extraoptions={},suppress_output=False):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-10
    options['num_repl_outer'] = 1
    options['num_repl_inner'] = 1
    options['max_iter'] = 100000
    options['outfolder'] = 'data/results/116_results'
    options['num_subjects'] = 100
    options['data_type'] = 'real'
    options['threads'] = 8
    options['HMM'] = False
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    os.makedirs(options['outfolder'],exist_ok=True)
    os.makedirs(options['outfolder']+'/params',exist_ok=True)
    p = 116
    K = options['K']
    ranks = np.arange(1,40,2).astype(int)
    options['experiment_name'] = 'realdata_'+options['modelname']+'_K='+str(K)
    options['LR'] = 0  

    try:
        df = pd.read_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
        num_done = df['inner'].max()+1
        if num_done==options['num_repl_outer']:
            print('Experiment already done')
            return
    except:
        df = pd.DataFrame()
        num_done = 0

    data_folder = 'data/processed/'
    if options['modelname'] in ['Complex_Watson','Complex_ACG']:
        add_complex = '_complex'
    else:
        add_complex = ''
    data_file = data_folder+'fMRI_SchaeferTian116_GSR'+add_complex+'.h5'
    with h5.File(data_file,'r') as f:
        data_train = f['U_train'][:][:1200*options['num_subjects']]
        L_train = f['L_train'][:][:1200*options['num_subjects']]
        data_test = f['U_test'][:][:1200*options['num_subjects']]
        L_test = f['L_test'][:][:1200*options['num_subjects']]

    if options['modelname'] in ['Watson','Complex_Watson','ACG','Complex_ACG']:
        data_train = data_train[:,:,0]
        data_test = data_test[:,:,0]

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

                params,df = run(data_train=data_train,L_train=L_train,data_test=data_test,L_test=L_test,K=K,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)
                np.save(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_K'+str(K)+'_params.npy',params)

            else:
                if options['modelname'] in ['Watson','Complex_Watson']:        
                    break
                print('Running model:',options['modelname'],'rank:',rank,'inner:',inner)
                options['init'] = 'no'

                params_MM = np.load(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(ranks[i-1])+'_K'+str(K)+'_params.npy',allow_pickle=True).item()

                params,df = run(data_train=data_train,L_train=L_train,data_test=data_test,L_test=L_test,K=K,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)
                np.save(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_K'+str(K)+'_params.npy',params)

if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        print(sys.argv)
        options = {}
        options['modelname'] = sys.argv[1]
        options['K'] = int(sys.argv[2])
        run_experiment(extraoptions=options,suppress_output=True)
    else:
        modelnames = ['Watson','ACG','MACG','SingularWishart']
        # modelnames = ['Complex_Watson']
        K = 1
        for modelname in modelnames:
            run_experiment(extraoptions={'modelname':modelname,'K':K},suppress_output=False)