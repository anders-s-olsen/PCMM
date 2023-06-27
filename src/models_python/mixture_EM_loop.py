import numpy as np
import tqdm

def mixture_EM_loop(dist,data,tol=1e-8,max_iter=10000,num_repl=1,init=None):

    best_loglik = -1000000

    for repl in range(num_repl):
        print(['Initializing repl '+str(repl)])
        if init is None or init == 'dc' or init == 'diametrical_clustering':
            dist.initialize(X=data,init='dc')
        elif init == '++' or init == 'plusplus' or init == 'diametrical_clustering_plusplus':
            dist.initialize(X=data,init='++')
        elif init == 'uniform' or init == 'unif':
            dist.initialize(init='uniform')
        loglik = []
        print('Beginning EM loop')
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
    from WatsonMixtureEM import Watson
    from ACGMixtureEM import ACG
    from MACGMixtureEM import MACG
    import matplotlib.pyplot as plt
    K = np.array(2)
    
    p = np.array(3)
    W = Watson(K=K,p=p)
    ACG = ACG(K=K,p=p)
    MACG = MACG(K=K,p=p,q=2)

    # data = np.loadtxt('data/synthetic/synth_data_4.csv',delimiter=',')
    # data1 = data[np.arange(2000,step=2),:]
    # data2 = np.zeros((1000,p,2))
    # data2[:,:,0] = data[np.arange(2000,step=2),:]
    # data2[:,:,1] = data[np.arange(2000,step=2)+1,:]

    # params_W,beta_W,loglik_W,_ = mixture_EM_loop(dist=W,data=data1,num_repl=3,init='unif')
    # params_ACG,beta_ACG,loglik_ACG,_ = mixture_EM_loop(dist=ACG,data=data1,num_repl=3,init='unif')
    # params_MACG,beta_MACG,loglik_MACG,_ = mixture_EM_loop(dist=MACG,data=data2,num_repl=3,init='unif')

    data = np.loadtxt('data/synthetic/synth_data_ACG.csv',delimiter=',')

    # params_W,beta_W,loglik_W,_ = mixture_EM_loop(dist=W,data=data,num_repl=3,init='unif')
    params_ACG,beta_ACG,loglik_ACG,_ = mixture_EM_loop(dist=ACG,data=data,num_repl=1,init='unif')
    # params_MACG,beta_MACG,loglik_MACG,_ = mixture_EM_loop(dist=MACG,data=data2,num_repl=3,init='unif')    
    
    stop=7