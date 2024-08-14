from src.helper_functions import calc_NMI,make_true_mat
import pandas as pd
import numpy as np
import os
import h5py as h5
from src.DMM_EM.riemannian_clustering import *
from scipy.cluster.vq import kmeans2

def run_experiment(extraoptions={},suppress_output=False):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-10
    options['num_repl_outer'] = 25
    options['num_repl_inner'] = 1
    options['max_iter'] = 100000
    options['outfolder'] = 'data/results/torchvsEM_phase_controlled_results'
    options['num_subjects'] = 1
    options['data_type'] = 'phase_controlled'
    options['threads'] = 8
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    os.makedirs(options['outfolder'],exist_ok=True)
    p = 116
    K = 5

    # load data using h5
    # with h5.File('data/synthetic/one_frequency_phase_controlled_116data_eida.h5','r') as f:
    # options['experiment_name'] = 'phase_amplitude_controlled_'+options['modelname']
    # with h5.File('data/synthetic/phase_amplitude_controlled_116data_eida.h5','r') as f:
    options['experiment_name'] = 'phase_controlled_'+options['modelname']
    with h5.File('data/synthetic/phase_controlled_116data_eida.h5','r') as f:

        data_train = f['U'][:]
        L_train = f['L'][:]
    if options['modelname']=='euclidean':
        data_train = data_train[:,:,0]
        data_train[np.sum(data_train,axis=-1)>p//2]=-data_train[np.sum(data_train,axis=-1)>p//2]
    elif options['modelname']=='diametrical':
        data_train = data_train[:,:,0]
    data_train = data_train[:1200*options['num_subjects']]
    L_train = L_train[:1200*options['num_subjects']]

    df = pd.DataFrame()
    P = np.double(make_true_mat(options['num_subjects']))

    options['init'] = '++'
    for inner in range(options['num_repl_outer']):
        print('Running experiment: ',options['experiment_name'],' inner: ',inner)
        if options['modelname']=='euclidean':
            _,labels = kmeans2(data_train,k=K,minit='++')
        elif options['modelname']=='diametrical':
            C,labels,obj = diametrical_clustering(data_train,K=K,max_iter=10000,num_repl=1,init=options['init'],tol=1e-16)
        elif options['modelname']=='grassmann':
            C,labels,obj = grassmann_clustering(data_train,K=K,max_iter=10000,num_repl=1,init=options['init'],tol=1e-16)
        elif options['modelname']=='weighted_grassmann':
            C,S,labels,obj = weighted_grassmann_clustering(data_train,L_train,K=K,max_iter=10000,num_repl=1,init=options['init'],tol=1e-16)
        labels = np.eye(K)[labels]
        labels = labels.T
        train_NMI = calc_NMI(P,np.double(np.array(labels)))
        entry = {'modelname':options['modelname'],'K':K,'p':p,'inner':inner,'train_NMI':train_NMI,'init':options['init']}
        df = pd.concat([df,pd.DataFrame([entry])],ignore_index=True)
        df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')

    options['init'] = 'unif'
    for inner in range(options['num_repl_outer']):
        print('Running experiment: ',options['experiment_name'],' inner: ',inner)
        if options['modelname']=='euclidean':
            init_matrix = np.random.random((K,p))
            init_matrix = init_matrix/np.linalg.norm(init_matrix,axis=-1)[:,np.newaxis]
            _,labels = kmeans2(data_train,k=init_matrix,minit='matrix')
        elif options['modelname']=='diametrical':
            C,labels,obj = diametrical_clustering(data_train,K=K,max_iter=10000,num_repl=1,init=options['init'],tol=1e-16)
        elif options['modelname']=='grassmann':
            C,labels,obj = grassmann_clustering(data_train,K=K,max_iter=10000,num_repl=1,init=options['init'],tol=1e-16)
        elif options['modelname']=='weighted_grassmann':
            C,S,labels,obj = weighted_grassmann_clustering(data_train,L_train,K=K,max_iter=10000,num_repl=1,init=options['init'],tol=1e-16)
        labels = np.eye(K)[labels]
        labels = labels.T
        train_NMI = calc_NMI(P,np.double(np.array(labels)))
        entry = {'modelname':options['modelname'],'K':K,'p':p,'inner':inner,'train_NMI':train_NMI,'init':options['init']}
        df = pd.concat([df,pd.DataFrame([entry])],ignore_index=True)
        df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')

if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        print(sys.argv)
        options = {}
        options['modelname'] = sys.argv[1]
        run_experiment(extraoptions=options,suppress_output=True)
    else:
        modelnames = ['euclidean','diametrical','grassmann','weighted_grassmann']
        # modelnames = ['diametrical']
        for modelname in modelnames:
            run_experiment(extraoptions={'modelname':modelname},suppress_output=False)