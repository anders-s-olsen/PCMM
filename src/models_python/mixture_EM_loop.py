import numpy as np
import tqdm

def mixture_EM_loop(model,data,tol=1e-8,max_iter=10000,num_repl=1,init=None):

    best_loglik = -1000000

    for repl in range(num_repl):
        print(['Initializing repl '+str(repl)])
        model.initialize(X=data,init=init,tol=tol)
        loglik = []
        print('Beginning EM loop')
        iter = 0
        while True:
        
            # E-step
            loglik.append(model.log_likelihood(X=data))
            if np.isnan(loglik[-1]):
                raise ValueError("Nan reached")

            if iter>1:
                if loglik[-1]-loglik[-2]<tol or iter==max_iter:
                    if loglik[-1]>best_loglik:
                        best_loglik = loglik[-1]
                        loglik_final = loglik
                        params_final = model.get_params()
                        beta_final = model.posterior(X=data)                 
                        num_iter_final = iter
                    break

            # M-step
            model.M_step(X=data,tol=tol)
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