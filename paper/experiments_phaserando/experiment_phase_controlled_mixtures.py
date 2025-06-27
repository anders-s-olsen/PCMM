from paper.helper_functions_paper import make_true_mat, run_phaserando
import pandas as pd
import numpy as np
import os
import h5py as h5

def run_experiment(extraoptions={},dataset='narrowband_phase_controlled',suppress_output=False):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-12
    options['num_repl_outer'] = 10
    options['max_iter'] = 100000
    options['outfolder'] = 'paper/data/results/torchvsEM_phase_controlled_results'
    options['num_subjects'] = 1
    options['num_repl'] = 1
    options['threads'] = 8
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    os.makedirs(options['outfolder'],exist_ok=True)
    p = 116
    K = 5
    ranks = [1,5,10,25,50]
    do_HMM = True
    P = np.double(make_true_mat(options['num_subjects'],K=K))

    data_folder = 'paper/data/phase_randomized/'
    options['experiment_name'] = dataset+'_'+options['modelname']+'_init'+options['experiment']
    data_file = data_folder+dataset+'_116data.h5'
    with h5.File(data_file,'r') as f:
        if options['modelname'] == 'Complex_Watson' or options['modelname'] == 'Complex_ACG' or options['modelname'] == 'complex_diametrical':
            data = f['data_complex_projective_hyperplane'][:]
        elif options['modelname'] == 'Watson' or options['modelname'] == 'ACG' or options['modelname'] == 'diametrical' or options['modelname'] == 'least_squares':
            data = f['data_real_projective_hyperplane'][:]
        elif options['modelname'] == 'MACG' or options['modelname'] == 'grassmann':
            data = f['data_grassmann'][:]
        elif options['modelname'] == 'SingularWishart' or options['modelname'] == 'weighted_grassmann':
            data = f['data_spsd'][:]
        elif options['modelname'] == 'Complex_Normal':
            data = f['data_analytic'][:]
        elif options['modelname'] == 'Normal':
            data = f['data_ts'][:]

    df = pd.DataFrame()
    num_done = 0

    data_train = data[:1200]
    data_test1 = data[1200:2400]
    data_test2 = data[2400:3600]

    for inner in range(num_done,options['num_repl_outer']):        
        for i,rank in enumerate(ranks):
            for LR in [0,0.1]: 
                options['LR'] = LR

                print('Running model:',options['modelname'],'rank:',rank,'LR:',LR,'inner:',inner)
                options['rank'] = rank
                if rank>1 and options['modelname'] in ['Watson','Complex_Watson']:
                    break

                if options['experiment']=='random':
                    options['init'] = 'unif'

                    params = None
                    options['HMM'] = False
                    _,df = run_phaserando(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)

                    if options['LR'] != 0: #rank 1 HMM
                        if do_HMM:
                            options['HMM'] = True
                            _,df = run_phaserando(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)
                elif options['experiment']=='Kmeans':
                    if options['modelname'] in ['Watson','Complex_Watson','ACG','Complex_ACG']:
                        options['init'] = 'dc' 
                    elif options['modelname']=='MACG':
                        options['init'] = 'gc'
                    elif options['modelname']=='SingularWishart':
                        options['init'] = 'wgc'
                    elif options['modelname'] in ['Normal','Complex_Normal']:
                        options['init'] = 'ls'
                    else:
                        raise ValueError("Problem")

                    params = None
                    options['HMM'] = False
                    _,df = run_phaserando(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)

                    if options['LR'] != 0: #rank 1 HMM
                        if do_HMM:
                            options['HMM'] = True
                            _,df = run_phaserando(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)
                elif options['experiment']=='Kmeansseg':
                    if options['modelname'] in ['Watson','Complex_Watson','ACG','Complex_ACG']:
                        options['init'] = 'dc_seg' #rank 1 model
                    elif options['modelname']=='MACG':
                        options['init'] = 'gc_seg'
                    elif options['modelname']=='SingularWishart':
                        options['init'] = 'wgc_seg'
                    elif options['modelname'] in ['Normal','Complex_Normal']:
                        options['init'] = 'ls_seg'
                    else:
                        raise ValueError("Problem")

                    params = None
                    options['HMM'] = False
                    _,df = run_phaserando(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)

                    if options['LR'] != 0: #rank 1 HMM
                        if do_HMM:
                            options['HMM'] = True
                            _,df = run_phaserando(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)
                elif options['experiment']=='ALL':
                    if options['modelname'] in ['Watson','Complex_Watson']:
                        continue
                    if i==0: #for the first rank, initialize with dc_seg for EM
                        print('Running model:',options['modelname'],'rank:',rank,'LR:',LR,'inner:',inner)

                        params = None
                        if options['modelname'] in ['Watson','Complex_Watson','ACG','Complex_ACG']:
                            options['init'] = 'dc_seg' #rank 1 model
                        elif options['modelname']=='MACG':
                            options['init'] = 'gc_seg'
                        elif options['modelname']=='SingularWishart':
                            options['init'] = 'wgc_seg'
                        elif options['modelname'] in ['Normal','Complex_Normal']:
                            options['init'] = 'ls_seg'
                        else:
                            raise ValueError("Problem")
                    else:
                        print('Running model:',options['modelname'],'rank:',rank,'LR:',LR,'inner:',inner)
                        options['init'] = 'no'

                        params = np.load(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(ranks[i-1])+'_LR'+str(LR)+'_params.npy',allow_pickle=True).item()

                    options['HMM'] = False
                    params,df = run_phaserando(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)
                    np.save(options['outfolder']+'/params/'+options['modelname']+'_rank'+str(rank)+'_LR'+str(LR)+'_params.npy',params)
                    if options['LR'] != 0: #rank X HMM
                        if do_HMM:
                            options['HMM'] = True
                            options['init'] = 'no'
                            _,df = run_phaserando(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,params=params,suppress_output=suppress_output,inner=inner,p=p)


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
        experiment = 'ALL'
        for modelname in modelnames:
            run_experiment(extraoptions={'modelname':modelname,'experiment':experiment},suppress_output=False)