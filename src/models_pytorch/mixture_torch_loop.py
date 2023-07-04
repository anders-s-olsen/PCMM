import numpy as np
import torch
import tqdm
torch.set_default_dtype(torch.float64)

def mixture_torch_loop(model,data,tol=1e-8,max_iter=10000,num_repl=1,init='no',LR=0.1):

    best_loglik = -1000000

    for repl in range(num_repl):
        print(['Initializing repl '+str(repl)])
        model.initialize(X=data,init=init,tol=tol)
        # the 'no'-option (default) is for ACG-lowrank, where some columns are randomly initialized and others prespecified
        
        optimizer = torch.optim.Adam(model.parameters(),lr=LR)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,threshold=tol,threshold_mode='abs',min_lr=0.0001,patience=100)
        scheduler = None

        loglik = []

        for iter in range(max_iter):
            
            epoch_nll = -model(data) #negative for nll

            optimizer.zero_grad(set_to_none=True)
            epoch_nll.backward()
            optimizer.step()
            loglik.append(-epoch_nll.item())

            if iter>100:
                if scheduler is not None:
                    scheduler.step(epoch_nll)
                    if optimizer.param_groups[0]["lr"]<0.001:
                        break
                else:
                    if loglik[-1]-loglik[-100]<tol:
                        break
            print('Done with iteration '+str(iter))
        
        if loglik[-1]>best_loglik:
            best_loglik = loglik[-1]
            loglik_final = loglik
            params_final = model.get_params()
            beta_final = model.posterior(X=data)            
            num_iter_final = iter


    return params_final,beta_final,loglik_final,num_iter_final

# def mixture_torch_loop_batch(model,data,tol=1e-8,max_iter=10000,num_repl=1,init='no'):

#     best_loglik = -1000000
#     n,_ = data.shape
#     batch_size = 64
#     num_batches = int(np.floor(n/batch_size))

#     for repl in range(num_repl):
#         print(['Initializing repl '+str(repl)])
#         model.initialize(X=data,init=init,tol=tol)
#         # the 'no'-option (default) is for ACG-lowrank, where some columns are randomly initialized and others prespecified
        
#         optimizer = torch.optim.Adam(model.parameters(),lr=1)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,threshold=tol,threshold_mode='abs',min_lr=0.0001,patience=100)
#         # scheduler = None

#         loglik = []

#         for iter in range(max_iter):
#             indices = torch.randperm(data.shape[0])
#             epoch_nll = 0
#             for batch in range(num_batches):
#                 idx = indices[:batch_size]
#                 indices = indices[batch_size:]
            
#                 batch_nll = -model(data[idx]) #negative for nll

#                 optimizer.zero_grad(set_to_none=True)
#                 batch_nll.backward()
#                 optimizer.step()
#                 epoch_nll += batch_nll
#             loglik.append(-epoch_nll.item())

#             if iter>100:
#                 if scheduler is not None:
#                     scheduler.step(epoch_nll)
#                     if optimizer.param_groups[0]["lr"]<0.001:
#                         break
#             print('Done with iteration '+str(iter))
        
#         if loglik[-1]>best_loglik:
#             best_loglik = loglik[-1]
#             loglik_final = loglik
#             params_final = model.get_params()
#             beta_final = model.posterior(X=data)            
#             num_iter_final = iter


#     return params_final,beta_final,loglik_final,num_iter_final

if __name__=='__main__':
    from Watson_torch import Watson
    from ACG_lowrank_torch import ACG
    # from MACGMixtureEM import MACG
    import matplotlib.pyplot as plt
    K = np.array(2)
    
    p = np.array(3)
    W = Watson(K=K,p=p)
    ACG = ACG(K=K,p=p)
    # MACG = MACG(K=K,p=p,q=2)

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
    params_ACG,beta_ACG,loglik_ACG,_ = mixture_torch_loop(dist=ACG,data=data,num_repl=1,init='unif')
    # params_MACG,beta_MACG,loglik_MACG,_ = mixture_EM_loop(dist=MACG,data=data2,num_repl=3,init='unif')    
    
    stop=7