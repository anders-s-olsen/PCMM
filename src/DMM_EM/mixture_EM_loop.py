import numpy as np
from tqdm import tqdm
from time import time
import os

def mixture_EM_loop(model,data,tol=1e-8,max_iter=10000,num_repl=1,init=None,suppress_output=False,threads=8):


    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(threads)

    best_loglik = -1000000

    for _ in range(num_repl):
        # print(['Initializing repl '+str(repl)])
        if init != 'no':
            model.initialize(X=data,init_method=init)
        loglik = []
        # print('Beginning EM loop')
        for epoch in tqdm(range(max_iter),disable=suppress_output):
        
            # E-step
            # time1 = time()
            loglik.append(model.log_likelihood(X=data))
            if np.isnan(loglik[-1]):
                raise ValueError("Nan reached")

            if epoch>5:
                latest = np.array(loglik[-5:])
                maxval = np.max(latest)
                secondhighest = np.max(latest[latest!=maxval])
                if np.abs((maxval-secondhighest)/maxval)<tol:
                    if loglik[-1]>best_loglik:
                        best_loglik = loglik[-1]
                        loglik_final = loglik
                        params_final = model.get_params()
                        beta_final = model.posterior(X=data)        
                    break

            # M-step
            # time2 = time()
            model.M_step(X=data)
            # time3 = time()
            # print('E-step time: '+str(time2-time1))
            # print('M-step time: '+str(time3-time2))
            # if epoch % 10 == 0:
            #     print(['Done with iteration '+str(epoch)])
    
    return params_final,beta_final,loglik_final

if __name__=='__main__':
    from WatsonEM import Watson
    from ACGEM import ACG
    from MACGEM import MACG
    import matplotlib.pyplot as plt
    K = 5
    p = 10

    for test in range(2,6):

        if test==1:
            W = Watson(K=K,p=p)
            data = np.loadtxt('data/synthetic/synth_data_ACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
            params,_,loglik = mixture_EM_loop(model=W,data=data,num_repl=1,init='unif')
            plt.figure()
            plt.plot(loglik)
            plt.title('Watson')
            plt.savefig('tmp/Watson_loglik.png')
            plt.close()
            plt.figure(figsize=(15,5))
            for k in range(K):
                plt.subplot(1,K,k+1)
                plt.barh(np.arange(p),params['mu'][:,k])
                # show only kappa and pi with first two decimals
                kappatitle = str(round(params['kappa'][k],2))
                pititle = str(round(params['pi'][k],2))
                plt.title('kappa='+kappatitle+', pi='+pititle)
            plt.savefig('tmp/Watson_params.png')
            plt.close()
        elif test==2:
            ACGobj = ACG(K=K,p=p,rank=2)
            data = np.loadtxt('data/synthetic/synth_data_ACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
            params,_,loglik = mixture_EM_loop(model=ACGobj,data=data,num_repl=1,init='unif')
            plt.figure()
            plt.plot(loglik)
            plt.title('ACG_lowrank')
            plt.savefig('tmp/ACG_lowrank_loglik.png')
            plt.close()
            plt.figure(figsize=(15,5))
            plt.subplots(1,K)
            for k in range(K):
                plt.subplot(1,K,k+1)
                L = params['M'][k]@params['M'][k].T+np.eye(p)
                plt.imshow(L)
                # plt.colorbar()
                plt.title('pi='+str(round(params['pi'][k],2)))
            plt.savefig('tmp/ACG_lowrank_params.png')
            plt.close()
        elif test==3:
            MACG_obj = MACG(K=K,p=p,q=2,rank=2)
            data = np.loadtxt('data/synthetic/synth_data_MACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
            data1 = data[np.arange(2000,step=2),:]
            data2 = np.zeros((1000,p,2))
            data2[:,:,0] = data[np.arange(2000,step=2),:]
            data2[:,:,1] = data[np.arange(2000,step=2)+1,:]
            params,_,loglik = mixture_EM_loop(model=MACG_obj,data=data2,num_repl=1,init='unif')
            plt.figure()
            plt.plot(loglik)
            plt.title('MACG_lowrank')
            plt.savefig('tmp/MACG_lowrank_loglik.png')
            plt.close()
            plt.figure(figsize=(15,5))
            plt.subplots(1,K)
            for k in range(K):
                plt.subplot(1,K,k+1)
                L = params['M'][k]@params['M'][k].T+np.eye(p)
                plt.imshow(L)
                # plt.colorbar()
                plt.title('pi='+str(round(params['pi'][k],2)))
            plt.savefig('tmp/MACG_lowrank_params.png')
            plt.close()
        elif test==4:
            ACGobj = ACG(K=K,p=p,rank=p)
            data = np.loadtxt('data/synthetic/synth_data_ACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
            params,_,loglik = mixture_EM_loop(model=ACGobj,data=data,num_repl=1,init='unif')
            plt.figure()
            plt.plot(loglik)
            plt.title('ACG_fullrank')
            plt.savefig('tmp/ACG_fullrank_loglik.png')
            plt.close()
            plt.figure(figsize=(15,5))
            plt.subplots(1,K)
            for k in range(K):
                plt.subplot(1,K,k+1)
                plt.imshow(params['Lambda'][k])
                # plt.colorbar()
                plt.title('pi='+str(round(params['pi'][k],2)))
            plt.savefig('tmp/ACG_fullrank_params.png')
            plt.close()
        elif test==5:
            MACG_obj = MACG(K=K,p=p,q=2,rank=p)
            data = np.loadtxt('data/synthetic/synth_data_MACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
            data1 = data[np.arange(2000,step=2),:]
            data2 = np.zeros((1000,p,2))
            data2[:,:,0] = data[np.arange(2000,step=2),:]
            data2[:,:,1] = data[np.arange(2000,step=2)+1,:]
            params,_,loglik = mixture_EM_loop(model=MACG_obj,data=data2,num_repl=1,init='unif')
            plt.figure()
            plt.plot(loglik)
            plt.title('MACG_fullrank')
            plt.savefig('tmp/MACG_fullrank_loglik.png')
            plt.close()
            plt.figure(figsize=(15,5))
            plt.subplots(1,K)
            for k in range(K):
                plt.subplot(1,K,k+1)
                plt.imshow(params['Lambda'][k])
                # plt.colorbar()
                plt.title('pi='+str(round(params['pi'][k],2)))
            plt.savefig('tmp/MACG_fullrank_params.png')
            plt.close()
