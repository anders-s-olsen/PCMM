import h5py
import numpy as np
import torch
from src.models_python import mixture_EM_loop, WatsonMixtureEM, ACGMixtureEM
from src.models_pytorch import MixtureModel_torch, Watson_torch, ACG_lowrank_torch
torch.set_num_threads(16)
import sys

def run_experiment(exp):
    ### load data, only the first 200 subjects (each with 1200 data points) (not the same subjects in train/test)
    data_train = np.array(h5py.File('data/processed/fMRI_atlas_RL2.h5', 'r')['Dataset'][:,:240000]).T
    n,p = data_train.shape
    print('Loaded training data')
    data_test = np.array(h5py.File('data/processed/fMRI_atlas_RL1.h5', 'r')['Dataset'][:,:240000]).T
    print('Loaded test data, beginning fit')

    if exp==0:
        name='Watson_EM'
    elif exp==1:
        name='ACG_EM'
    elif exp==2:
        name='Watson_torch'
    elif exp==3:
        name='ACG_torch'

    num_repl = 5

    ### EM algorithms
    if exp==0 or exp==1: #Watson EM
        for K in range(2,21):
            print('starting K='+str(K))
            if exp==0:
                model = WatsonMixtureEM.Watson(K=K,p=p)
            elif exp==1:
                model = ACGMixtureEM.ACG(K=K,p=p)

            # train
            params,_,loglik,num_iter = mixture_EM_loop.mixture_EM_loop(model,data_train,tol=1e-6,max_iter=100000,num_repl=num_repl,init='uniform')
            np.savetxt('experiments/outputs/'+name+'_454_trainlikelihoodcurve_K='+str(K)+'.csv',np.array(loglik))
            if exp ==0:
                np.savetxt('experiments/outputs/'+name+'_454_mu_K='+str(K)+'.csv',params['mu'])
                np.savetxt('experiments/outputs/'+name+'_454_kappa_K='+str(K)+'.csv',params['kappa'])
            elif exp == 1:
                for k in range(K):
                    np.savetxt('experiments/outputs/'+name+'_454_L_K='+str(K)+'_k'+str(k)+'.csv',params['Lambda'][k])
            np.savetxt('experiments/outputs/'+name+'_454_pi_K='+str(K)+'.csv',params['pi'])
            # test
            if exp==0:
                model = WatsonMixtureEM.Watson(K=K,p=p,params=params)
            elif exp==1:
                model = ACGMixtureEM.ACG(K=K,p=p,params=params)
            test_loglik = model.log_likelihood(X=data_test)
            np.savetxt('experiments/outputs/'+name+'_454_traintestlikelihood'+str(K)+'.csv',np.array([loglik[-1],test_loglik]))

    ### torch algorithms
    if exp==2 or exp==3: #Watson EM
        for K in range(2,21):
            print('starting K='+str(K))
            best_nll = 1000000
            for repl in range(num_repl):
                if exp==2:
                    model = MixtureModel_torch.TorchMixtureModel(Watson_torch.Watson,K=K,p=p)
                elif exp==3:
                    model = MixtureModel_torch.TorchMixtureModel(ACG_lowrank_torch.ACG,K=K,p=p,D=p)
                data = torch.tensor(data_train)
                optimizer = torch.optim.Adam(model.parameters(),lr=0.1)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,threshold=1e-6,threshold_mode='abs',min_lr=0.0001,patience=100)
    
                num_epoch = 100000
                epoch_nll_collector = []

                for epoch in range(num_epoch):
                    
                    epoch_nll = -model(data) #negative for nll

                    optimizer.zero_grad(set_to_none=True)
                    epoch_nll.backward()
                    optimizer.step()
                    epoch_nll_collector.append(epoch_nll.detach())

                    if epoch>100:
                        if scheduler is not None:
                            scheduler.step(epoch_nll)
                            if optimizer.param_groups[0]["lr"]<0.001:
                                break
                if epoch_nll<best_nll:
                    best_model = model
                    best_nll = epoch_nll

            params = best_model.get_model_param()
            np.savetxt('experiments/outputs/'+name+'_454_trainlikelihoodcurve_K='+str(K)+'.csv',np.array(epoch_nll_collector))
            LogSoftMax = torch.nn.LogSoftmax(dim=0)
            if exp ==2:
                mu = np.zeros((p,K))
                kappa = np.zeros(K)
                pi = np.zeros(K)
                SoftPlus = torch.nn.Softplus(beta=20, threshold=1)
                for k in range(K):
                    mu[:,k] = params['mix_comp_'+str(k)]['mu']
                    kappa[k] = params['mix_comp_'+str(k)]['kappa']
                    pi[k] = params['mix_comp_'+str(k)]['pi']
                np.savetxt('experiments/outputs/'+name+'_454_mu_K='+str(K)+'.csv',torch.nn.functional.normalize(mu,dim=0))
                np.savetxt('experiments/outputs/'+name+'_454_kappa_K='+str(K)+'.csv',SoftPlus(kappa))
                np.savetxt('experiments/outputs/'+name+'_454_pi_K='+str(K)+'.csv',LogSoftMax(pi))
                
            elif exp == 3:
                pi = np.zeros(K)
                for k in range(K):
                    L_tri_inv = params['mix_comp_'+str(k)]['L_tri_inv']
                    L = torch.linalg.inv(L_tri_inv@L_tri_inv.T)
                    L = p*L/torch.trace(L)
                    pi[k] = params['mix_comp_'+str(k)]['pi']
                np.savetxt('experiments/outputs/'+name+'_454_L_K='+str(K)+'k'+str(k)+'.csv',L)
                np.savetxt('experiments/outputs/'+name+'_454_pi_K='+str(K)+'.csv',LogSoftMax(pi))

            test_loglik = best_model.forward(X=data_test)

            np.savetxt('experiments/outputs/'+name+'_454_traintestlikelihood'+str(K)+'.csv',np.array([best_nll[-1],test_loglik]))



if __name__=="__main__":
    # run_experiment(exp=int(0))
    run_experiment(exp=int(sys.argv[1]))