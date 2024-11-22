from PCMM.helper_functions import train_model,calc_NMI,make_true_mat,test_model
import pandas as pd
import numpy as np
import torch
import os
import h5py as h5

def run(data_train,data_test1,data_test2,K,P,df,options,params=None,suppress_output=False,inner=None,p=116):
    params,train_posterior,loglik_curve = train_model(data_train,K=K,options=options,suppress_output=suppress_output,samples_per_sequence=1200,params=params)
    test1_loglik,test1_posterior,_ = test_model(data_test=data_test1,params=params,K=K,options=options,samples_per_sequence=1200)
    test2_loglik,test2_posterior,_ = test_model(data_test=data_test2,params=params,K=K,options=options,samples_per_sequence=1200)
    train_NMI = calc_NMI(P,np.double(np.array(train_posterior)))
    test1_NMI = calc_NMI(P,np.double(np.array(test1_posterior)))
    test2_NMI = calc_NMI(P,np.double(np.array(test2_posterior)))
    entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),'K':K,'p':p,'rank':options['rank'],'inner':inner,'iter':len(loglik_curve),
             'train_loglik':loglik_curve[-1],'test1_loglik':test1_loglik,'test2_loglik':test2_loglik,
             'train_NMI':train_NMI,'test1_NMI':test1_NMI,'test2_NMI':test2_NMI}
    df = pd.concat([df,pd.DataFrame([entry])],ignore_index=True)
    # df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
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
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    os.makedirs(options['outfolder'],exist_ok=True)
    p = 116
    K = 5
    ranks = [1,5,10]
    do_HMM = True
    P = np.double(make_true_mat(options['num_subjects'],K=K))

    data_folder = 'data/synthetic/'
    options['experiment_name'] = dataset+'_'+options['modelname']+'_init'+options['experiment']
    data_file = data_folder+dataset+'_116data_eida.h5'
    with h5.File(data_file,'r') as f:
        if 'complex' in options['modelname'] or 'Complex' in options['modelname']:
            data = f['U_complex'][:][:,:,0]
        elif options['modelname'] in ['Watson','ACG','kmeans']:
            data = f['U'][:,:,0]
        elif options['modelname'] in ['grassmann','MACG']:
            data = f['U'][:]
        elif options['modelname'] in ['weighted_grassmann','SingularWishart']:
            data = f['U'][:]*np.sqrt(f['L'][:])[:,None,:]
        elif options['modelname'] in ['Normal']:
            data = f['S'][:][:,:,0]

    try:
        df = pd.read_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
        num_done = df['inner'].max()
        df = df[df['inner']!=num_done]
    except:
        df = pd.DataFrame()
        num_done = 0

    data_train = data[:1200]
    data_test1 = data[1200:2400]
    data_test2 = data[2400:3600]

    for inner in range(num_done,options['num_repl_outer']):        
        for LR in [0,0.1]: 
            options['LR'] = LR

            for i,rank in enumerate(ranks):
                print('Running model:',options['modelname'],'rank:',rank,'LR:',LR,'inner:',inner)
                options['rank'] = rank
                if rank>1 and options['modelname'] in ['Watson','Complex_Watson']:
                    break

                if options['experiment']=='random':
                    options['init'] = 'unif'

                    params = None
                    options['HMM'] = False
                    _,df = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)

                    if options['LR'] != 0: #rank 1 HMM
                        if do_HMM:
                            options['HMM'] = True
                            _,df = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)
                elif options['experiment']=='Kmeans':
                    if options['modelname'] in ['Watson','Complex_Watson','ACG','Complex_ACG']:
                        options['init'] = 'dc' 
                    elif options['modelname']=='MACG':
                        options['init'] = 'gc'
                    elif options['modelname']=='SingularWishart':
                        options['init'] = 'wgc'
                    elif options['modelname']=='Normal':
                        options['init'] = 'euclidean'
                    else:
                        raise ValueError("Problem")

                    params = None
                    options['HMM'] = False
                    _,df = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)

                    if options['LR'] != 0: #rank 1 HMM
                        if do_HMM:
                            options['HMM'] = True
                            _,df = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)
                elif options['experiment']=='Kmeansseg':
                    if options['modelname'] in ['Watson','Complex_Watson','ACG','Complex_ACG']:
                        options['init'] = 'dc_seg' #rank 1 model
                    elif options['modelname']=='MACG':
                        options['init'] = 'gc_seg'
                    elif options['modelname']=='SingularWishart':
                        options['init'] = 'wgc_seg'
                    elif options['modelname']=='Normal':
                        options['init'] = 'euclidean_seg'
                    else:
                        raise ValueError("Problem")

                    params = None
                    options['HMM'] = False
                    _,df = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)

                    if options['LR'] != 0: #rank 1 HMM
                        if do_HMM:
                            options['HMM'] = True
                            _,df = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)
                elif options['experiment']=='ALL':
                    if i==0: #for the first rank, initialize with dc_seg for EM
                        print('Running model:',options['modelname'],'rank:',rank,'LR:',LR,'inner:',inner)

                        if LR==0: #restart initialization
                            params = None
                            if options['modelname'] in ['Watson','Complex_Watson','ACG','Complex_ACG']:
                                options['init'] = 'dc_seg' #rank 1 model
                            elif options['modelname']=='MACG':
                                options['init'] = 'gc_seg'
                            elif options['modelname']=='SingularWishart':
                                options['init'] = 'wgc_seg'
                            elif options['modelname']=='Normal':
                                options['init'] = 'euclidean_seg'
                            else:
                                raise ValueError("Problem")

                        else: #load EM model
                            params = np.load(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_params.npy',allow_pickle=True).item()
                            for key in params.keys():
                                params[key] = torch.tensor(params[key])
                            options['init'] = 'no'

                        options['HMM'] = False
                        params,df = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)
                        if LR==0:
                            np.save(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_params.npy',params)

                        if options['LR'] != 0: #rank 1 HMM
                            if do_HMM:
                                options['init'] = 'no'
                                options['HMM'] = True
                                _,df = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)
                    else:
                        print('Running model:',options['modelname'],'rank:',rank,'LR:',LR,'inner:',inner)
                        options['init'] = 'no'
                        options['HMM'] = False

                        if LR == 0: #load previous rank
                            params = np.load(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(ranks[i-1])+'_params.npy',allow_pickle=True).item()
                        else:
                            params = np.load(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_params.npy',allow_pickle=True).item()
                            for key in params.keys():
                                params[key] = torch.tensor(params[key])
                                
                        params,df = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)
                        if LR==0:
                            np.save(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_params.npy',params)

                        if options['LR'] != 0: #rank X HMM
                            if do_HMM:
                                options['HMM'] = True
                                _,df = run(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)


if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        print(sys.argv)
        options = {}
        options['modelname'] = sys.argv[1]
        options['experiment'] = sys.argv[3]
        run_experiment(extraoptions=options,dataset=sys.argv[2],suppress_output=True)
    else:
        modelnames = ['Watson','Complex_Watson','ACG','Complex_ACG','MACG','SingularWishart']
        modelnames = ['Complex_ACG']
        experiment = 'Kmeansseg'
        for modelname in modelnames:
            run_experiment(extraoptions={'modelname':modelname,'experiment':experiment},dataset='phase_narrowband_controlled',suppress_output=False)