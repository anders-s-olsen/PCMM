from src.helper_functions import train_model
import numpy as np
import h5py
import time

def run_experiment(extraoptions={},suppress_output=False):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-8
    options['num_repl_outer'] = 1
    options['num_repl_inner'] = 1
    options['max_iter'] = 5
    options['outfolder'] = 'data/results/EM_memorytest'
    options['num_subjects'] = 10
    options['ACG_rank'] = 'full' #for ACG and MACG
    options['data_type'] = 'fullres'
    options['threads'] = 8
    options['LR'] = 0
    options['init'] = '++'
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    options['experiment_name'] = 'EM_memorytest'
    K = options['K']

    subjects = ['100206','100307','100408']

    print(options)
    print('loading data...')
    time1 = time.time()
    options['ACG_rank'] = 'lowrank'
    if options['modelname']=='MACG':
        data_train = np.array(h5py.File('data/processed/fMRI_full_RL1.h5','r')['Dataset'][:,:options['num_subjects']*1200*2]).T
    else:
        data_train = np.array(h5py.File('data/processed/fMRI_full_RL1.h5','r')['Dataset'][:,np.arange(options['num_subjects']*1200*2,step=2)]).T
    time2 = time.time()
    print('loaded data in '+str(time2-time1)+' seconds')
    print('starting model...')
    params,train_posterior,loglik_curve = train_model(data_train=data_train,K=K,options=options,suppress_output=suppress_output)
    time3 = time.time()
    print('Done in '+str(time3-time2)+' seconds')


if __name__=="__main__":

    import sys
    if len(sys.argv)>1:
        print(sys.argv)
        options = {}
        options['modelname'] = sys.argv[1]
        options['K'] = sys.argv[2]
        run_experiment(extraoptions=options,suppress_output=False)
    else:
        run_experiment(extraoptions={'modelname':'Watson','K':2},suppress_output=False)
