import numpy as np
import torch
from tqdm import tqdm

def mixture_torch_loop(model,data,L=None,tol=1e-8,max_iter=100000,num_repl=1,init=None,LR=0.1,suppress_output=False,threads=8):
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)
    best_loglik = -1000000

    for repl in range(num_repl):
        # print(['Initializing inner repl '+str(repl)])
        if init != 'no':
            model.initialize(X=data,L=L,init_method=init)

        if model.distribution in ['SingularWishart_fullrank','SingularWishart_lowrank','SingularWishart']:
            X = data*np.sqrt(L[:,None,:])
        else:
            X = data
        if model.HMM:
            if init not in ['unif','uniform']:
                model.initialize_transition_matrix(X=X)

        optimizer = torch.optim.Adam(model.parameters(),lr=LR)
        # tol = 0.0001
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',threshold=tol,threshold_mode='rel',min_lr=0.0001,patience=3)
        scheduler = None
        loglik = []
        params = []
        print('Beginning numerical optimization loop')

        for epoch in tqdm(range(max_iter),disable=suppress_output):
            epoch_nll = -model(X) #negative for nll

            if torch.isnan(-epoch_nll):
                raise ValueError("Nan reached")
            
            optimizer.zero_grad(set_to_none=True)
            epoch_nll.backward()
            optimizer.step()
            loglik.append(-epoch_nll.item())

            # get state dict
            params.append(model.get_params())
            if len(params)>5:
                params.pop(0)
            
            if epoch>5:
                if scheduler is not None:
                    scheduler.step(epoch_nll)
                    if optimizer.param_groups[0]["lr"]<0.001:
                        break
                else:
                    latest = np.array(loglik[-5:])
                    maxval = np.max(latest)
                    secondhighest = np.max(latest[latest!=maxval])
                    if (maxval-secondhighest)/np.abs(maxval)<tol or latest[-1]==np.min(latest):
                        optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]*0.1
                        if optimizer.param_groups[0]["lr"]<0.001:
                            if loglik[-1]>best_loglik:
                                best_loglik = loglik[-1]
                                loglik_final = loglik
                                params_final = params[np.where(loglik[-5:]==maxval)[0].item()]
                                model.set_params(params_final)
                                beta_final = model.posterior(X=X)         
                            break
    if 'params_final' not in locals():
        params_final = model.get_params()
        beta_final = model.posterior(X=X)
        loglik_final = loglik
        
    return params_final,beta_final,loglik_final