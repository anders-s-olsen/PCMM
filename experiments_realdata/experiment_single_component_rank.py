from src.helper_functions import train_model,test_model,load_fMRI_data,calc_ll_per_sub
import pandas as pd
import numpy as np
import os
import h5py as h5

def run(data_train,data_test1,data_test2,K,df,options,params=None,suppress_output=False,inner=None,p=116):
    params,train_posterior,loglik_curve = train_model(data_train,K=K,options=options,suppress_output=suppress_output,samples_per_sequence=1200,params=params)
    
    train_loglik,train_posterior,train_loglik_per_sample = test_model(data_test=data_train,params=params,K=K,options=options)
    test1_loglik,test1_posterior,test1_loglik_per_sample = test_model(data_test=data_test1,params=params,K=K,options=options)
    test2_loglik,test2_posterior,test2_loglik_per_sample = test_model(data_test=data_test2,params=params,K=K,options=options)

    train_ll,test1_ll,test2_ll = calc_ll_per_sub(train_loglik_per_sample,test1_loglik_per_sample,test2_loglik_per_sample)
    np.savetxt(options['outfolder']+'/lls/'+options['experiment_name']+'_rank='+str(options['rank'])+'_train_ll'+'_inner'+str(inner)+'.txt',train_ll)
    np.savetxt(options['outfolder']+'/lls/'+options['experiment_name']+'_rank='+str(options['rank'])+'_test1_ll'+'_inner'+str(inner)+'.txt',test1_ll)
    np.savetxt(options['outfolder']+'/lls/'+options['experiment_name']+'_rank='+str(options['rank'])+'_test2_ll'+'_inner'+str(inner)+'.txt',test2_ll)

    entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),
             'K':K,'p':p,'rank':options['rank'],'inner':inner,'iter':len(loglik_curve),
             'train_loglik':loglik_curve[-1],'test1_loglik':test1_loglik,'test2_loglik':test2_loglik}
    #save the posterior as txt
    if K!=1:
        np.savetxt(options['outfolder']+'/posteriors/'+options['experiment_name']+'_train_posterior'+'_inner'+str(inner)+'.txt',train_posterior)
        np.savetxt(options['outfolder']+'/posteriors/'+options['experiment_name']+'_test1_posterior'+'_inner'+str(inner)+'.txt',test1_posterior)
        np.savetxt(options['outfolder']+'/posteriors/'+options['experiment_name']+'_test2_posterior'+'_inner'+str(inner)+'.txt',test2_posterior)
    df = pd.concat([df,pd.DataFrame([entry])],ignore_index=True)
    df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
    return params,df

def run_experiment(extraoptions={},suppress_output=False):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-10
    options['num_repl_outer'] = 10
    options['num_repl_inner'] = 1
    options['max_iter'] = 100000
    options['outfolder'] = 'data/results/116_results'
    options['data_type'] = 'real'
    options['threads'] = 8
    options['HMM'] = False
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    os.makedirs(options['outfolder'],exist_ok=True)
    os.makedirs(options['outfolder']+'/params',exist_ok=True)

    # set parameters
    p = 116
    K = options['K']
    ranks = np.arange(1,40,2).astype(int)
    options['experiment_name'] = options['dataset']+'rank_realdata_'+options['modelname']+'_K='+str(K)
    options['LR'] = 0  

    # check if the experiment is already done
    try:
        df = pd.read_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
        num_done = df['inner'].max()
        if num_done==options['num_repl_outer']:
            print('Experiment already done')
            return
        df = df[df['inner']!=num_done]
    except:
        df = pd.DataFrame()
        num_done = 0

    data_folder = 'data/processed/'
    data_file = data_folder+options['dataset']+'fMRI_SchaeferTian116_GSR.h5'
    data_train,data_test1,data_test2 = load_fMRI_data(data_file,options,only_some_points=False)

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
                elif options['modelname'] in ['Normal']:
                    options['init'] = 'euclidean'
                elif options['modelname'] in ['Complex_Normal']:
                    options['init'] = 'unif'
                else:
                    options['init'] = '++'

                params,df = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)
                np.save(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_K'+str(K)+'_params.npy',params)

            else:
                if options['modelname'] in ['Watson','Complex_Watson']:        
                    break
                print('Running model:',options['modelname'],'rank:',rank,'inner:',inner)
                options['init'] = 'no'

                params_MM = np.load(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(ranks[i-1])+'_K'+str(K)+'_params.npy',allow_pickle=True).item()
                params,df = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)
                np.save(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_K'+str(K)+'_params.npy',params)

if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        print(sys.argv)
        options = {}
        options['modelname'] = sys.argv[1]
        options['K'] = int(sys.argv[2])
        options['dataset'] = sys.argv[3]
        run_experiment(extraoptions=options,suppress_output=True)
    else:
        modelnames = ['Watson','ACG','MACG','SingularWishart']
        dataset = 'REST1REST2' # 'REST' or 'MOTOR' or 'SOCIAL'
        modelnames = ['Complex_ACG']
        K = 1
        for modelname in modelnames:
            run_experiment(extraoptions={'modelname':modelname,'K':K,'dataset':dataset},suppress_output=False)