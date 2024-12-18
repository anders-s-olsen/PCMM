import numpy as np
from tqdm import tqdm
from copy import deepcopy

def mixture_EM_loop(model,data,tol=1e-8,max_iter=10000,num_repl=1,init=None,suppress_output=False):

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
                    model.M = model.init_M_svd_given_M_init(X=data,M_init=model2.M,beta=beta)
                elif model.distribution in ['SingularWishart_lowrank','Normal_lowrank','Complex_Normal_lowrank']:
                    model.M = model.init_M_svd_given_M_init(X=data,M_init=model2.M,beta=beta,gamma=model2.gamma)

        loglik = []
        params = []
        print('Beginning EM loop')
        pbar = tqdm(total=max_iter,disable=suppress_output)
        for epoch in range(max_iter):
        
            # E-step
            loglik.append(model.log_likelihood(X=data))
            if np.isnan(loglik[-1]):
                raise ValueError("Nan reached. There can be many possible reasons for this, including the initialization of the model, the data, or the model itself. First try reinitializing the same model.")
            
            params.append(model.get_params())
            #remove first entry in params
            if len(params)>10:
                params.pop(0)

            if epoch>10:
                latest = np.array(loglik[-10:])
                maxval = np.max(latest)
                try:
                    secondhighest = np.max(latest[latest!=maxval])
                    crit = (maxval-secondhighest)/np.abs(maxval)
                    if crit<tol or latest[-1]==np.min(latest):
                        done=True
                except:
                    crit = tol
                    done=True
                pbar.set_description('Convergence towards tol: %.2e'%crit)
                pbar.update(1)
                if done:
                    if maxval>best_loglik:
                        best_loglik = maxval
                        loglik_final = loglik
                        best = np.where(loglik[-10:]==maxval)[0]
                        if hasattr(best,"__len__")>0: # in the rare case of two equal values....
                            best = best[0]
                        params_final = params[best.item()]
                        model2 = deepcopy(model)
                        model2.set_params(params_final)
                        beta_final = model2.posterior(X=data)
                    break
            else:
                pbar.set_description('In the initial phase')
                pbar.update(1)

            # M-step
            model.M_step(X=data)
            
    # if no params_final variable exists
    if 'params_final' not in locals():
        params_final = model.get_params()
        beta_final = model.posterior(X=data)
        loglik_final = loglik
    
    return params_final,beta_final,loglik_final