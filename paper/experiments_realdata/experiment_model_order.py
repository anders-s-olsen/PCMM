from PCMM.helper_functions import train_model,test_model,calc_ll_per_sub
import pandas as pd
import os
import numpy as np
import h5py as h5
def load_fMRI_data(data_file,options,only_some_points=False):
    assert options['modelname'] in ['Watson','ACG','MACG','SingularWishart','Complex_Watson','Complex_ACG','Normal','Complex_Normal','euclidean','diametrical','complex_diametrical','grassmann','weighted_grassmann']

    with h5.File(data_file,'r') as f:
        if options['modelname'] in ['Complex_Watson','Complex_ACG','complex_diametrical']:
            # complex normalized phase vectors
            data_train = f['U_complex_train'][:][:,:,0]
            data_test1 = f['U_complex_test1'][:][:,:,0]
            data_test2 = f['U_complex_test2'][:][:,:,0] 
        elif options['modelname'] in ['Watson','ACG','euclidean','diametrical']:
            # leading eigenvector of cosinus phase coherence matrix
            data_train = f['U_cos_train'][:][:,:,0]
            data_test1 = f['U_cos_test1'][:][:,:,0]
            data_test2 = f['U_cos_test2'][:][:,:,0]
        elif options['modelname'] in ['MACG','grassmann']:
            # both eigenvectors of cosinus phase coherence matrix
            data_train = f['U_cos_train'][:]
            data_test1 = f['U_cos_test1'][:]
            data_test2 = f['U_cos_test2'][:]
        elif options['modelname'] in ['SingularWishart','weighted_grassmann']:
            # both eigenvectors and both eigenvalues of cosinus phase coherence matrix
            data_train = f['U_cos_train'][:]*np.sqrt(f['L_cos_train'][:][:,None,:])
            data_test1 = f['U_cos_test1'][:]*np.sqrt(f['L_cos_test1'][:][:,None,:])
            data_test2 = f['U_cos_test2'][:]*np.sqrt(f['L_cos_test2'][:][:,None,:])
        elif options['modelname'] in ['Complex_Normal']:
            # complex normalized phase vectors scaled by hilbert amplitude
            data_train = f['U_complex_train'][:][:,:,0]*f['A_train'][:]
            data_test1 = f['U_complex_test1'][:][:,:,0]*f['A_test1'][:]
            data_test2 = f['U_complex_test2'][:][:,:,0]*f['A_test2'][:]
        elif options['modelname'] in ['Normal']:
            # filtered time series data (no Hilbert transform)
            data_train = f['timeseries_train'][:][:,:,0]
            data_test1 = f['timeseries_test1'][:][:,:,0]
            data_test2 = f['timeseries_test2'][:][:,:,0]
        else:
            raise ValueError("Problem")
    if only_some_points:
        data_train = data_train[:1000]
        data_test1 = data_test1[:1000]
        data_test2 = data_test2[:1000]
    return data_train,data_test1,data_test2

def run(data_train,data_test1,data_test2,K,df,options,params=None,suppress_output=False,inner=None,p=116):
    params,train_posterior,loglik_curve = train_model(data_train,K=K,options=options,suppress_output=suppress_output,samples_per_sequence=1200,params=params)
    
    train_loglik,train_posterior,train_loglik_per_sample = test_model(data_test=data_train,params=params,K=K,options=options,samples_per_sequence=1200)
    test1_loglik,test1_posterior,test1_loglik_per_sample = test_model(data_test=data_test1,params=params,K=K,options=options,samples_per_sequence=1200)
    test2_loglik,test2_posterior,test2_loglik_per_sample = test_model(data_test=data_test2,params=params,K=K,options=options,samples_per_sequence=1200)

    train_ll,test1_ll,test2_ll = calc_ll_per_sub(train_loglik_per_sample,test1_loglik_per_sample,test2_loglik_per_sample)
    np.savetxt(options['outfolder']+'/lls/'+options['experiment_name']+'_train_ll'+'_inner'+str(inner)+'.txt',train_ll)
    np.savetxt(options['outfolder']+'/lls/'+options['experiment_name']+'_test1_ll'+'_inner'+str(inner)+'.txt',test1_ll)
    np.savetxt(options['outfolder']+'/lls/'+options['experiment_name']+'_test2_ll'+'_inner'+str(inner)+'.txt',test2_ll)

    entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),
             'K':K,'p':p,'rank':options['rank'],'inner':inner,'iter':len(loglik_curve),
             'train_loglik':loglik_curve[-1],'test1_loglik':test1_loglik,'test2_loglik':test2_loglik}
    #save the posterior as txt
    if K!=1:
        np.savetxt(options['outfolder']+'/posteriors/'+options['experiment_name']+'_train_posterior'+'_inner'+str(inner)+'.txt',train_posterior)
        np.savetxt(options['outfolder']+'/posteriors/'+options['experiment_name']+'_test1_posterior'+'_inner'+str(inner)+'.txt',test1_posterior)
        np.savetxt(options['outfolder']+'/posteriors/'+options['experiment_name']+'_test2_posterior'+'_inner'+str(inner)+'.txt',test2_posterior)
    df = pd.concat([df,pd.DataFrame([entry])],ignore_index=True)
    return params,df

def run_experiment(extraoptions={},suppress_output=False,save_params=True,GSR='GSR'):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-10
    options['num_repl_outer'] = 10
    options['num_repl_inner'] = 1
    options['max_iter'] = 100000
    options['outfolder'] = 'paper/data/results/116_results'
    options['HMM'] = False
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
    options['LR'] = 0
    # check if the experiment is already done
    try:
        df = pd.read_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
        num_done = len(df)
        if num_done==options['num_repl_outer']:
            print('Experiment already done')
            return
    except:
        df = pd.DataFrame()
        num_done = 0

    data_train,data_test1,data_test2 = load_fMRI_data(data_file,options,only_some_points=False)

    for inner in range(num_done,options['num_repl_outer']):   
        print('Running model:',options['modelname'],'rank:',options['rank'],'inner:',inner)

        params_MM = None
        if options['modelname'] in ['Watson','ACG','Complex_Watson','Complex_ACG']:
            options['init'] = 'dc' #rank 1 model
        elif options['modelname']=='MACG':
            options['init'] = 'gc'
        elif options['modelname']=='SingularWishart':
            options['init'] = 'wgc'
        elif options['modelname'] in ['Normal']:
            options['init'] = 'ls_seg'
        else:
            options['init'] = '++'

        params,df = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,df=df,options=options,params=params_MM,suppress_output=suppress_output,inner=inner,p=p)
        # if save_params:
            # print('Saving params but not dataframe')
        # np.save(options['outfolder']+'/params/'+options['experiment_name']+'_inner'+str(inner)+'.npy',params)
        # else:
        df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')

if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        print(sys.argv)
        options = {}
        options['modelname'] = sys.argv[1]
        options['K'] = int(sys.argv[2])
        options['rank'] = int(sys.argv[3])
        options['dataset'] = sys.argv[4]
        GSR = sys.argv[5]
        run_experiment(extraoptions=options,suppress_output=True,save_params=False,GSR=GSR)
    else:
        modelnames = ['Watson','ACG','MACG','SingularWishart','Complex_Watson','Complex_ACG','Normal','Complex_Normal','euclidean','diametrical','complex_diametrical','grassmann','weighted_grassmann']
        modelnames = ['Complex_ACG']
        K = 2
        dataset =  'MOTORSOCIAL' # 'MOTORSOCIAL', 'REST1REST2'
        rank = 1
        # rank = 'fullrank'
        for modelname in modelnames:
            run_experiment(extraoptions={'modelname':modelname,'K':K,'rank':rank,'dataset':dataset},suppress_output=False,save_params=True,GSR='GSR')