import numpy as np
import h5py
import torch
import os
from src.models_python.ACGMixtureEM import ACG as ACG_EM
from src.models_python.ACGLowrankEM import ACG as ACG_lowrank_EM
from src.models_python.MACGMixtureEM import MACG as MACG_EM
from src.models_python.MACGLowrankEM import MACG as MACG_lowrank_EM
from src.models_python.WatsonMixtureEM import Watson as Watson_EM
from src.models_python.mixture_EM_loop import mixture_EM_loop
from src.models_pytorch.ACG_lowrank_torch import ACG as ACG_torch
from src.models_pytorch.MACG_lowrank_torch import MACG as MACG_torch
from src.models_pytorch.Watson_torch import Watson as Watson_torch
from src.models_pytorch.mixture_torch_loop import mixture_torch_loop
from src.models_python.diametrical_clustering import diametrical_clustering, diametrical_clustering_plusplus
from src.models_pytorch.diametrical_clustering_torch import diametrical_clustering_torch, diametrical_clustering_plusplus_torch
def load_data(options,p=3,K=2):

    if options['data_type']=='fMRI_SchaeferTian454':
        loc = 'data/processed/fMRI_SchaeferTian454_RL'
    elif options['data_type']=='fMRI_full':
        loc = 'data/processed/fMRI_full_RL'
    elif options['data_type']=='fMRI_SchaeferTian116':
        loc = 'data/processed/fMRI_SchaeferTian116_RL'
    elif options['data_type']=='fMRI_SchaeferTian116_GSR':
        loc = 'data/processed/fMRI_SchaeferTian116_GSR_RL'
    elif options['data_type']=='synth' or options['data_type']=='synthetic':
        loc = 'data/synthetic/synth_data_'

    if options['modelname']=='Watson' or options['modelname']=='ACG' or options['modelname']=='ACG_lowrank':
        num_eigs=1
    elif options['modelname']=='MACG' or options['modelname']=='MACG_lowrank':
        num_eigs=2
    
    if options['data_type']!='synth' and options['data_type']!='synthetic':
        data_train_tmp = np.array(h5py.File(loc+'2.h5', 'r')['Dataset'][:,:options['num_subjects']*1200*2]).T
        data_test_tmp = np.array(h5py.File(loc+'1.h5', 'r')['Dataset'][:,:options['num_subjects']*1200*2]).T
        data_test2_tmp = np.array(h5py.File(loc+'2.h5', 'r')['Dataset'][:,options['num_subjects']*1200*2:options['num_subjects']*1200*4]).T
        if num_eigs==1:
            data_train = data_train_tmp[np.arange(options['num_subjects']*1200*2,step=2),:]
            data_test = data_test_tmp[np.arange(options['num_subjects']*1200*2,step=2),:]
            data_test2 = data_test2_tmp[np.arange(options['num_subjects']*1200*2,step=2),:]
        elif num_eigs == 2:
            p = data_train_tmp.shape[1]
            data_train = np.zeros((options['num_subjects']*1200,p,2))
            data_train[:,:,0] = data_train_tmp[np.arange(options['num_subjects']*1200*2,step=2),:]
            data_train[:,:,1] = data_train_tmp[np.arange(options['num_subjects']*1200*2,step=2)+1,:]
            data_test = np.zeros((options['num_subjects']*1200,p,2))
            data_test[:,:,0] = data_test_tmp[np.arange(options['num_subjects']*1200*2,step=2),:]
            data_test[:,:,1] = data_test_tmp[np.arange(options['num_subjects']*1200*2,step=2)+1,:]
            data_test2 = np.zeros((options['num_subjects']*1200,p,2))
            data_test2[:,:,0] = data_test2_tmp[np.arange(options['num_subjects']*1200*2,step=2),:]
            data_test2[:,:,1] = data_test2_tmp[np.arange(options['num_subjects']*1200*2,step=2)+1,:]
    elif options['data_type']=='synth' or options['data_type']=='synthetic':
        if num_eigs==1:
            data_train = np.loadtxt('data/synthetic/synth_data_ACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
            data_test = np.loadtxt('data/synthetic/synth_data_ACG_p'+str(p)+'K'+str(K)+'_2.csv',delimiter=',')
        elif num_eigs==2:
            data_train = np.zeros((1000,p,2))
            data_train_tmp = np.loadtxt('data/synthetic/synth_data_MACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
            data_train[:,:,0] = data_train_tmp[np.arange(2000,step=2),:]
            data_train[:,:,1] = data_train_tmp[np.arange(2000,step=2)+1,:]
            data_test = np.zeros((1000,p,2))
            data_test_tmp = np.loadtxt('data/synthetic/synth_data_MACG_p'+str(p)+'K'+str(K)+'_2.csv',delimiter=',')
            data_test[:,:,0] = data_test_tmp[np.arange(2000,step=2),:]
            data_test[:,:,1] = data_test_tmp[np.arange(2000,step=2)+1,:]
        data_test2 = data_test

    if options['LR']!=0:
        data_train = torch.tensor(data_train)
        data_test = torch.tensor(data_test)
        data_test2 = torch.tensor(data_test2)
    return data_train,data_test,data_test2

def train_model(data_train,K,options,params=None):
    torch.set_default_dtype(torch.float64)
    if options['LR']==0:
        os.environ["OMP_NUM_THREADS"] = str(options['threads'])
    else:
        os.environ["OMP_NUM_THREADS"] = '1'
        torch.set_num_threads(options['threads'])
        # torch.set_num_interop_threads(options['threads'])

    p = data_train.shape[1]
    if options['ACG_rank']=='full':
        rank=p
    else: rank=options['ACG_rank']
    if options['modelname'] == 'Watson':
        if options['LR']==0:
            model = Watson_EM(K=K,p=p,params=params)
        else:
            model = Watson_torch(K=K,p=p,params=params)
    elif options['modelname'] == 'ACG':
        if options['LR']==0:
            model = ACG_EM(K=K,p=p,params=params)
        else:
            model = ACG_torch(K=K,p=p,rank=rank,params=params) #cholesky formulation when full rank
    elif options['modelname'] == 'ACG_lowrank':
        if options['LR']==0:
            model = ACG_lowrank_EM(K=K,p=p,rank=rank,params=params)
        else:
            model = ACG_torch(K=K,p=p,rank=rank,params=params) #cholesky formulation when full rank            
    elif options['modelname'] == 'MACG':
        if options['LR']==0:
            model = MACG_EM(K=K,p=p,q=2,params=params)
        else:
            model = MACG_torch(K=K,p=p,q=2,rank=rank,params=params)
    elif options['modelname'] == 'MACG_lowrank':
        if options['LR']==0:
            model = MACG_lowrank_EM(K=K,p=p,q=2,rank=rank,params=params)
        else:
            model = MACG_torch(K=K,p=p,q=2,rank=rank,params=params)   
        
    
    if options['LR']==0: #EM
        params,_,loglik,_ = mixture_EM_loop(model,data_train,tol=options['tol'],max_iter=options['max_iter'],
                                                num_repl=options['num_repl_inner'],init=options['init'])
    else:
        params,_,loglik,_ = mixture_torch_loop(model,data_train,tol=options['tol'],max_iter=options['max_iter'],
                                        num_repl=options['num_repl_inner'],init=options['init'],LR=options['LR'])
    
    return params,loglik[-1],loglik
    
def test_model(data_test,params,K,options):
    p = data_test.shape[1]
    if options['ACG_rank']=='full':
        rank=p
    else: rank=options['ACG_rank']
    if options['LR'] == 0:
        if options['modelname'] == 'Watson':    
            model = Watson_EM(K=K,p=p,params=params)
        elif options['modelname'] == 'ACG':
            model = ACG_EM(K=K,p=p,params=params)
        elif options['modelname'] == 'ACG_lowrank':
            model = ACG_EM(K=K,p=p,rank=rank,params=params)
        elif options['modelname'] == 'MACG':
            model = MACG_EM(K=K,p=p,q=2,params=params)
        test_loglik = model.log_likelihood(X=data_test)
        params_transformed = model.get_params()
    else:
        Softmax = torch.nn.Softmax(dim=0)
        pi_soft = Softmax(params['pi'])
        if options['modelname'] == 'Watson':
            mu_norm = torch.nn.functional.normalize(params['mu'],dim=0)
            kappa = params['kappa']
            params_transformed={'mu':mu_norm,'kappa':kappa,'pi':pi_soft}
        else:
            M = params['M']
            params_transformed={'M':M,'pi':pi_soft}
        with torch.no_grad():
            if options['modelname'] == 'Watson':
                model = Watson_torch(K=K,p=p,params=params_transformed)
            elif options['modelname'] == 'ACG':
                model = ACG_torch(K=K,p=p,rank=rank,params=params_transformed) 
            elif options['modelname'] == 'MACG':
                model = MACG_torch(K=K,p=p,q=2,rank=rank,params=params_transformed) 
            test_loglik = model.test_log_likelihood(X=data_test)  
    return test_loglik,params_transformed

def run_model_reps_and_save_logliks(data_train,data_test,data_test2,K,options):
    os.makedirs(options['outfolder'],exist_ok=True)
    rep_order = np.arange(options['num_repl_outer'])
    np.random.shuffle(rep_order)
    try:
        logliks = np.loadtxt(options['outfolder']+'/'+options['modelname']+'_'+options['experiment_name']+'_traintestlikelihood.csv',logliks)
    except:
        logliks = np.zeros((3,options['num_repl_outer']))
            

    for repl in range(options['num_repl_outer']):
        rep = rep_order[repl]
        if logliks[0,rep]==0:
            print('starting K='+str(K)+' rep='+str(rep))
            params,train_loglik,_ = train_model(data_train=data_train,K=K,options=options)
            test_loglik,_ = test_model(data_test=data_test,params=params,K=K,options=options)
            test_loglik2,_ = test_model(data_test=data_test2,params=params,K=K,options=options)
            logliks[0,rep] = train_loglik
            logliks[1,rep] = test_loglik
            logliks[2,rep] = test_loglik2
            np.savetxt(options['outfolder']+'/'+options['modelname']+'_'+options['experiment_name']+'_traintestlikelihood.csv',logliks)
        
def parse_input_args(args):
    options = {}
    options['modelname'] = args[1]
    options['LR'] = float(args[2])
    options['init'] = args[3]
    if len(args)>4:
        options['GSR'] = int(args[4])
    else:
        options['GSR'] = None
    return options


