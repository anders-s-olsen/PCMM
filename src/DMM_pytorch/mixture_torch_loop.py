import numpy as np
import torch
from tqdm import tqdm
from src.DMM_EM.DMMEMBaseModel import DMMEMBaseModel

def mixture_torch_loop(model,data,L=None,tol=1e-8,max_iter=100000,num_repl=1,init=None,LR=0.1,suppress_output=False,threads=8):
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)
    best_loglik = -1000000

    for repl in range(num_repl):
        # print(['Initializing inner repl '+str(repl)])
        if init != 'no':
            model.initialize(X=data,L=L,init_method=init)

        if model.distribution == 'SingularWishart_lowrank':
            X = data*np.sqrt(L[:,None,:])
        else:
            X = data
        if model.HMM:
            if init not in ['unif','uniform']:
                model.initialize_transition_matrix(X=X)

        if model.distribution in ['ACG_lowrank','MACG_lowrank','SingularWishart_lowrank']:
            if model.M.shape[-1]!=model.r:
                model2 = model.clone()
                model2.r = model2.M.shape[-1]
                Beta = model2.posterior(X=X)
                M = DMMEMBaseModel.init_M_svd_given_M_init(X=X.numpy(),M_init=model.M.detach().numpy(),Beta=Beta.numpy())
                model.M = torch.nn.Parameter(torch.tensor(M))

        optimizer = torch.optim.Adam(model.parameters(),lr=LR)
        loglik = []
        params = []
        done = False
        print('Beginning numerical optimization loop')

        pbar = tqdm(total=max_iter,disable=suppress_output)

        for epoch in range(max_iter):
            epoch_nll = -model(X) #negative for nll

            if torch.isnan(-epoch_nll):
                raise ValueError("Nan reached")
            
            optimizer.zero_grad(set_to_none=True)
            epoch_nll.backward()
            optimizer.step()
            loglik.append(-epoch_nll.item())

            # get state dict
            params.append(model.get_params())
            if len(params)>10:
                params.pop(0)
            
            if len(params)==10:
                latest = np.array(loglik[-10:])
                maxval = np.max(latest)
                try:
                    secondhighest = np.max(latest[latest!=maxval])
                except:
                    secondhighest = maxval
                crit = (maxval-secondhighest)/np.abs(maxval)
                if crit<tol or latest[-1]==np.min(latest):
                    if optimizer.param_groups[0]['lr']<1e-1:
                        done = True
                    else:
                        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
                        print('Learning rate reduced to:',optimizer.param_groups[0]['lr'],'after',epoch,'iterations')
                        params = []
                pbar.set_description('Convergence towards tol: %.2e'%crit)
                # pbar.set_postfix({'Epoch':epoch})
                pbar.update(1)
                if done:
                    if loglik[-1]>best_loglik:
                        best_loglik = loglik[-1]
                        loglik_final = loglik
                        best = np.where(loglik[-10:]==maxval)[0]
                        if hasattr(best,"__len__")>0: # in the rare case of two equal values....
                            best = best[0]
                        params_final = params[best]
                        model.set_params(params_final)
                        beta_final = model.posterior(X=X)         
                    break
            else:
                pbar.set_description('In the initial phase')
                pbar.update(1)
    if 'params_final' not in locals():
        params_final = model.get_params()
        beta_final = model.posterior(X=X)
        loglik_final = loglik
        
    return params_final,beta_final,loglik_final