import numpy as np
from tqdm import tqdm
from copy import deepcopy
from PCMM.PCMMnumpyBaseModel import init_M_svd_given_M_init

def mixture_EM_loop(model,data,tol=1e-8,max_iter=10000,num_repl=1,init=None,suppress_output=False, num_comparison=10):

    best_loglik = -1000000
    if 'Complex' in model.distribution:
        # check if data is also complex
        if not np.iscomplexobj(data):
            raise ValueError('Data must be complex for complex models')
    else:
        if np.iscomplexobj(data):
            raise ValueError('Data must be real for real models')

    for repl in range(num_repl):
        done = False
        # print(['Initializing repl '+str(repl)])
        if init != 'no':
            if 'pi' in model.__dict__:
                raise ValueError('Model already initialized, please set params=None or init=''no''')
            model.initialize(X=data,init_method=init) #NB using 'data', not 'X'
        else:
            if 'pi' not in model.__dict__:
                raise ValueError('Model not initialized, please provide an initialization method or a set of parameters')

        if 'lowrank' in model.distribution:
            if model.M.shape[-1]!=model.r:
                model2 = deepcopy(model)
                model2.r = model2.M.shape[-1]
                beta = model2.posterior(X=data)
                if model.distribution in ['ACG_lowrank','Complex_ACG_lowrank','MACG_lowrank']:
                    model.M = init_M_svd_given_M_init(X=data,K=model.K,r=model.r,M_init=model2.M,beta=beta,gamma=None,distribution=model.distribution)
                elif model.distribution in ['SingularWishart_lowrank','Normal_lowrank','Complex_Normal_lowrank']:
                    model.M = init_M_svd_given_M_init(X=data,K=model.K,r=model.r,M_init=model2.M,beta=beta,gamma=model2.gamma,distribution=model.distribution)

        loglik = []
        best_epoch_loglik = -np.inf
        params = []
        print('Beginning EM loop')
        pbar = tqdm(total=max_iter,disable=suppress_output)
        pbar.set_description('In the initial phase')
        pbar.update(0)
        for epoch in range(max_iter):
        
            # E-step
            loglik.append(model.log_likelihood(X=data))
            if np.isnan(loglik[-1]):
                raise ValueError("Nan reached. There can be many possible reasons for this, including the initialization of the model, the data, or the model itself. First try reinitializing the same model.")
            
            params.append(model.get_params())
            if loglik[-1]>best_epoch_loglik:
                best_model_params = deepcopy(model.get_params())
                best_epoch_loglik = loglik[-1]
            if epoch>num_comparison:
                latest = np.array(loglik[-num_comparison:])
                maxval = np.max(latest)
                try:
                    secondhighest = np.max(latest[latest!=maxval])
                    crit = (maxval-secondhighest)/np.abs(maxval)
                    if crit<tol or latest[-1]==np.min(latest):
                        done=True
                except:
                    crit = tol
                    done=True
                pbar.set_description('Loglik: %.2f, relative change: %.2e'%(loglik[-1],crit))
                pbar.update(1)
                if done:
                    if best_epoch_loglik>best_loglik:
                        best_loglik = best_epoch_loglik
                        params_final = best_model_params
                        loglik_final = loglik
                        model.set_params(best_model_params)
                        beta_final = model.posterior(X=data)
                    break
            else:
                pbar.set_description('Loglik: %.2f: '%loglik[-1])
                pbar.update(1)

            # M-step
            model.M_step(X=data)
            
    # if no params_final variable exists
    if 'params_final' not in locals():
        params_final = model.get_params()
        beta_final = model.posterior(X=data)
        loglik_final = loglik
    
    return params_final,beta_final,loglik_final