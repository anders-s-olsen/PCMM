from src.helper_functions import train_model,calc_NMI,make_true_mat
import pandas as pd
import numpy as np
import os
import h5py as h5
from src.DMM_EM.riemannian_clustering import *
from scipy.cluster.vq import kmeans2

def run_experiment(extraoptions={},suppress_output=False):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-8
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

    options['experiment_name'] = 'phase_controlled_'+options['modelname']
    df = pd.DataFrame()
    P = np.double(make_true_mat(options['num_subjects']))

    options['init'] = '++'
    for inner in range(options['num_repl_outer']):
        print('Running experiment: ',options['experiment_name'],' inner: ',inner)
        if options['modelname']=='euclidean':
            _,labels = kmeans2(data_train,K,minit='++')
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
            _,labels = kmeans2(data_train,K,minit='random')
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
    modelnames = ['euclidean','diametrical','grassmann','weighted_grassmann']
    for modelname in modelnames:
        run_experiment(extraoptions={'modelname':modelname},suppress_output=False)
    # run_experiment(extraoptions={'modelname':'Watson','LR':0},suppress_output=False)
