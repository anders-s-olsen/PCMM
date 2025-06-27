from paper.helper_functions_paper import run_phaserando,make_true_mat
import pandas as pd
import numpy as np
import os
import h5py as h5
from PCMM.phase_coherence_kmeans import *

def run_experiment(extraoptions={},dataset='phase_controlled',suppress_output=False):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-10
    options['num_repl_outer'] = 100
    options['max_iter'] = 100000
    options['outfolder'] = 'paper/data/results/torchvsEM_phase_controlled_results'
    options['num_subjects'] = 1
    options['HMM'] = False
    options['num_repl'] = 1
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    os.makedirs(options['outfolder'],exist_ok=True)
    p = 116
    K = 5
    P = np.double(make_true_mat(options['num_subjects'],K=K))

    data_folder = 'paper/data/phase_randomized/'
    options['experiment_name'] = dataset+'_'+options['modelname']
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

    data_train = data[:1200]
    data_test1 = data[1200:2400]
    data_test2 = data[2400:3600]
    df = pd.DataFrame()

    P = np.double(make_true_mat(options['num_subjects']))

    options['init'] = '++'
    for inner in range(options['num_repl_outer']):
        print('Running experiment: ',options['experiment_name'],' inner: ',inner)
        _,df = run_phaserando(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,suppress_output=suppress_output,inner=inner,p=p)

    options['init'] = 'uniform'
    for inner in range(options['num_repl_outer']):
        print('Running experiment: ',options['experiment_name'],' inner: ',inner)
        _,df = run_phaserando(data_train=data_train,data_test1=data_test1,data_test2=data_test2,K=K,P=P,df=df,options=options,suppress_output=suppress_output,inner=inner,p=p)

if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        print(sys.argv)
        options = {}
        options['modelname'] = sys.argv[1]
        run_experiment(extraoptions=options,dataset=sys.argv[2],suppress_output=True)
    else:
        modelnames = ['least_squares','diametrical','complex_diametrical','grassmann','weighted_grassmann']
        # modelnames = ['complex_diametrical']
        for modelname in modelnames:
            run_experiment(extraoptions={'modelname':modelname},dataset='narrowband_phase_controlled',suppress_output=False)