import numpy as np
from WatsonMixtureEM import Watson

def mixture_EM_loop(dist,data,tol=1e-8,max_iter=1000,num_repl=1,init=None):

    best_loglik = -1000000

    for repl in range(num_repl):
        if init is None or init == 'dc' or init == 'diametrical_clustering':
            dist.initialize(X=data,init='dc')
        elif init == '++' or init == 'plusplus' or init == 'diametrical_clustering_plusplus':
            dist.initialize(X=data,init='++')
        elif init == 'uniform' or init == 'unif':
            dist.initialize(init='uniform')
        loglik = []
        iter = 0
        while True:
        
            # E-step
            loglik.append(dist.log_likelihood(X=data))

            if iter>0:
                if abs(loglik[-2]-loglik[-1])<tol or iter==max_iter:
                    if loglik[-1]>best_loglik:
                        best_loglik = loglik[-1]
                        loglik_final = loglik
                        params_final = dist.get_parameters()
                        beta_final = np.exp(dist.density-dist.logsum_density).T                 
                        num_iter_final = iter
                    break

            # M-step
            dist.M_step(X=data)
            print(['Done with iteration '+str(iter)])
            iter +=1
    
    return params_final,beta_final,loglik_final,num_iter_final

if __name__=='__main__':
    import matplotlib.pyplot as plt
    K = np.array(2)
    
    p = np.array(3)
    W = Watson(K=K,p=p)

    data = np.loadtxt('synth_data_2.csv',delimiter=',')
    data = data[np.arange(2000,step=2),:]

    params,beta,loglik,_ = mixture_EM_loop(dist=W,data=data,num_repl=1)
    
    stop=7