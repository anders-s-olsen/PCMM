import numpy as np
import torch
from tqdm import tqdm
from PCMM.PCMMnumpyBaseModel import init_M_svd_given_M_init
from copy import deepcopy

def mixture_torch_loop(model,data,tol=1e-8,max_iter=100000,num_repl=1,init=None,LR=0.1,suppress_output=False,threads=8,decrease_lr_on_plateau=False,num_comparison=50):
    torch.set_num_threads(threads)
    best_loglik = -np.inf

    if 'lowrank' in model.distribution:
        assert model.r is not None, 'Model rank must be set'
        assert model.r!=0, 'Model rank must be non-zero'

    #check if data is a torch tensor
    if not isinstance(data,torch.Tensor):
        data = torch.from_numpy(data)
    
    if 'Complex' in model.distribution:
        # check if data is also complex
        if not data.is_complex():
            raise ValueError('Data must be complex for complex models')
    else:
        if data.is_complex():
            raise ValueError('Data must be real for real models')

    param_names = [name for name, _ in model.named_parameters()]
    if init == 'no' and 'pi' not in param_names:
        raise ValueError('Model not initialized, please provide an initialization method or a set of parameters')

    for repl in range(num_repl):
        if decrease_lr_on_plateau:
            flag_already_decreased_lr = False
        # print(['Initializing inner repl '+str(repl)])
        if init != 'no':
            # Each replication is an independent fit and therefore needs a
            # fresh initialization, including when a previous replication has
            # already registered parameters on the model.
            model.initialize(X=data,init_method=init)
        model.to(device=data.device)
        param_names = [name for name, _ in model.named_parameters()]

        if 'lowrank' in model.distribution:
            if model.M.shape[-1]!=model.r:
                model2 = deepcopy(model)
                model2.r = model2.M.shape[-1]
                beta = model2.posterior(X=data)
                if model.distribution in ['ACG_lowrank','Complex_ACG_lowrank','MACG_lowrank']:
                    gamma = None
                elif model.distribution in ['SingularWishart_lowrank','Normal_lowrank','Complex_Normal_lowrank']:
                    gamma = model.gamma.detach().cpu().numpy()
                M = init_M_svd_given_M_init(
                    X=data.detach().cpu().numpy(),
                    K=model.K,
                    r=model.r,
                    M_init=model.M.detach().cpu().numpy(),
                    beta=beta,
                    gamma=gamma,
                    distribution=model.distribution,
                )
                model.M = torch.nn.Parameter(torch.as_tensor(M, dtype=data.dtype, device=data.device))
            
        if model.HMM:
            # A newly initialized replication must not retain the transition
            # matrix fitted or initialized by the preceding replication.
            if init != 'no' or 'T' not in param_names:
                if init in ['unif','uniform']:
                    model.T = torch.nn.Parameter(
                        torch.full(
                            (model.K, model.K),
                            1 / model.K,
                            dtype=data.real.dtype,
                            device=data.device,
                        )
                    )
                else:
                    T,delta = model.initialize_transition_matrix_hmm(X=data)
                    model.T = torch.nn.Parameter(T)
                    model.pi = torch.nn.Parameter(delta)

        optimizer = torch.optim.Adam(model.parameters(),lr=LR)
        loglik = []
        best_epoch_loglik = -np.inf
        done = False
        if not suppress_output:
            tqdm.write('Beginning numerical optimization loop')

        pbar = tqdm(total=max_iter,disable=suppress_output)
        pbar.set_description('In the initial phase')
        pbar.update(0)

        for epoch in range(max_iter):
            epoch_nll = -model(data) #negative for nll

            if torch.isnan(-epoch_nll):
                raise ValueError("Nan reached")
            
            optimizer.zero_grad(set_to_none=True)
            epoch_nll.backward()
            loglik.append(-epoch_nll.item())

            with torch.no_grad():
                if loglik[-1]>best_epoch_loglik:
                    best_model_params = deepcopy(model.get_params())
                    best_epoch_loglik = loglik[-1]

            optimizer.step()

            with torch.no_grad():
                if epoch>=num_comparison:
                    latest = np.array(loglik[-num_comparison:])
                    maxval = np.max(latest)
                    try:
                        secondhighest = np.max(latest[latest!=maxval])
                    except ValueError:
                        secondhighest = maxval
                    crit = (maxval-secondhighest)/np.abs(maxval)
                    pbar.set_description('Loglik: %.2f, relative change: %.2e'%(loglik[-1],crit))
                    pbar.update(1)
                    if crit<tol or latest[-1]==np.min(latest):
                        if decrease_lr_on_plateau:
                            if flag_already_decreased_lr:
                                done = True
                            else:                       
                                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
                                if not suppress_output:
                                    tqdm.write(
                                        'Learning rate reduced to: %s after %d iterations'
                                        % (optimizer.param_groups[0]['lr'], epoch)
                                    )
                                flag_already_decreased_lr = True
                        else:
                            done = True
                    if done:
                        break
                else:
                    pbar.set_description('Loglik: %.2f: '%loglik[-1])
                    pbar.update(1)
        pbar.close()

        if best_epoch_loglik > best_loglik:
            best_loglik = best_epoch_loglik
            params_final = best_model_params
            loglik_final = loglik

    with torch.no_grad():
        model.set_params(params_final)
        beta_final = model.posterior(X=data)
        
    return params_final,beta_final,loglik_final
