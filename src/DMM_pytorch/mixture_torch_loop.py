import numpy as np
import torch
from tqdm import tqdm

def mixture_torch_loop(model,data,tol=1e-8,max_iter=100000,num_repl=1,init=None,LR=0.1,suppress_output=False,threads=8):
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)
    best_loglik = -1000000

    for repl in range(num_repl):
        # print(['Initializing inner repl '+str(repl)])
        if init != 'no':
            model.initialize(X=data,init_method=init)
        optimizer = torch.optim.Adam(model.parameters(),lr=LR)
        # optimizer = torch.optim.SGD(model.parameters(),lr=LR)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,threshold=tol,threshold_mode='abs',min_lr=0.0001,patience=100)
        scheduler = None

        loglik = []
        # print('Beginning optimization loop')

        for epoch in tqdm(range(max_iter),disable=suppress_output):
            epoch_nll = -model(data) #negative for nll

            if torch.isnan(-epoch_nll):
                raise ValueError("Nan reached")
            
            optimizer.zero_grad(set_to_none=True)
            epoch_nll.backward()
            optimizer.step()
            loglik.append(-epoch_nll.item())
            
            if epoch>5:
                if scheduler is not None:
                    scheduler.step(epoch_nll)
                    if optimizer.param_groups[0]["lr"]<0.001:
                        break
                else:
                    latest = np.array(loglik[-5:])
                    maxval = np.max(latest)
                    secondhighest = np.max(latest[latest!=maxval])
                    if np.abs((maxval-secondhighest)/maxval)<tol or latest[-1]==np.min(latest):
                        if loglik[-1]>best_loglik:
                            best_loglik = loglik[-1]
                            loglik_final = loglik
                            params_final = model.get_params()
                            beta_final = model.posterior(X=data)    
                        break
    if 'params_final' not in locals():
        params_final = model.get_params()
        beta_final = model.posterior(X=data)
        loglik_final = loglik
        
    return params_final,beta_final,loglik_final

if __name__=='__main__':
    from WatsonPyTorch import Watson
    from ACGPyTorch import ACG
    from MACGPyTorch import MACG
    import matplotlib.pyplot as plt
    K = 5
    p = 10

    for test in range(2,4):
        if test==1:
            W = Watson(K=K,p=p)
            data = np.loadtxt('data/synthetic/synth_data_ACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
            data = torch.tensor(data)
            params,_,loglik = mixture_torch_loop(model=W,data=data,num_repl=1,init='unif')
            plt.figure()
            plt.plot(loglik)
            plt.title('Watson')
            plt.savefig('tmp/Watson_loglik_pt.png')
            plt.close()
            plt.figure(figsize=(15,5))
            for k in range(K):
                plt.subplot(1,K,k+1)
                plt.barh(np.arange(p),np.array(params['mu'][:,k]))
                # show only kappa and pi with first two decimals
                kappatitle = str(round(params['kappa'][k].item(),2))
                pititle = str(round(params['pi'][k].item(),2))
                plt.title('kappa='+kappatitle+', pi='+pititle)
            plt.savefig('tmp/Watson_params_pt.png')
            plt.close()
        elif test==2:
            ACGobj = ACG(K=K,p=p,rank=2,HMM=True)
            data = np.loadtxt('data/synthetic/synth_data_ACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
            data = torch.tensor(data)
            params,_,loglik = mixture_torch_loop(model=ACGobj,data=data,num_repl=1,init='unif')
            plt.figure()
            plt.plot(loglik)
            plt.title('ACG_lowrank')
            plt.savefig('tmp/ACG_lowrank_loglik_pt.png')
            plt.close()
            plt.figure(figsize=(15,5))
            plt.subplots(1,K)
            for k in range(K):
                plt.subplot(1,K,k+1)
                L = params['M'][k]@params['M'][k].T+np.eye(p)
                plt.imshow(L)
                # plt.colorbar()
                plt.title('pi='+str(round(params['pi'][k].item(),2)))
            plt.savefig('tmp/ACG_lowrank_params_pt.png')
            plt.close()
        elif test==3:
            MACG_obj = MACG(K=K,p=p,q=2,rank=2)
            data = np.loadtxt('data/synthetic/synth_data_MACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
            data1 = data[np.arange(2000,step=2),:]
            data2 = np.zeros((1000,p,2))
            data2[:,:,0] = data[np.arange(2000,step=2),:]
            data2[:,:,1] = data[np.arange(2000,step=2)+1,:]
            data2 = torch.tensor(data2)
            params,_,loglik = mixture_torch_loop(model=MACG_obj,data=data2,num_repl=1,init='unif')
            plt.figure()
            plt.plot(loglik)
            plt.title('MACG_lowrank')
            plt.savefig('tmp/MACG_lowrank_loglik_pt.png')
            plt.close()
            plt.figure(figsize=(15,5))
            plt.subplots(1,K)
            for k in range(K):
                plt.subplot(1,K,k+1)
                L = params['M'][k]@params['M'][k].T+np.eye(p)
                plt.imshow(L)
                # plt.colorbar()
                plt.title('pi='+str(round(params['pi'][k].item(),2)))
            plt.savefig('tmp/MACG_lowrank_params_pt.png')
            plt.close()
