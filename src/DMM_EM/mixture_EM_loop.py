import numpy as np
from tqdm import tqdm
from time import time
import os

def mixture_EM_loop(model,data,L=None,tol=1e-8,max_iter=10000,num_repl=1,init=None,suppress_output=False,threads=8):

    best_loglik = -1000000

    for repl in range(num_repl):
        done = False
        # print(['Initializing repl '+str(repl)])
        if init != 'no':
            model.initialize(X=data,L=L,init_method=init)
        
        if model.distribution in ['SingularWishart_fullrank','SingularWishart_lowrank','SingularWishart']:
            X = data*np.sqrt(L[:,None,:])
        else:
            X = data

        loglik = []
        params = []
        print('Beginning EM loop')
        for epoch in tqdm(range(max_iter),disable=suppress_output):
        
            # E-step
            # time1 = time()
            loglik.append(model.log_likelihood(X=X))
            if np.isnan(loglik[-1]):
                raise ValueError("Nan reached")
            
            params.append(model.get_params())
            #remove first entry in params
            if len(params)>5:
                params.pop(0)

            if epoch>5:
                latest = np.array(loglik[-5:])
                maxval = np.max(latest)
                try:
                    secondhighest = np.max(latest[latest!=maxval])
                    if (maxval-secondhighest)/np.abs(maxval)<tol or latest[-1]==np.min(latest):
                        done=True
                except:
                    done=True
                if done:
                    if maxval>best_loglik:
                        best_loglik = maxval
                        loglik_final = loglik
                        best = np.where(loglik[-5:]==maxval)[0]
                        if hasattr(best,"__len__")>0: # in the rare case of two equal values....
                            best = best[0]
                        params_final = params[best.item()]
                        model.set_params(params_final)
                        beta_final = model.posterior(X=X)        
                    break

            # M-step
            # time2 = time()
            model.M_step(X=X)
            # np.savetxt('tmp/progress.txt','iter '+str(epoch)+' ll:'+str(loglik[-1]))
            # time3 = time()
            # print('E-step time: '+str(time2-time1))
            # print('M-step time: '+str(time3-time2))
            # if epoch % 10 == 0:
            #     print(['Done with iteration '+str(epoch)])
    # if no params_final variable exists
    if 'params_final' not in locals():
        params_final = model.get_params()
        beta_final = model.posterior(X=X)
        loglik_final = loglik
    
    return params_final,beta_final,loglik_final