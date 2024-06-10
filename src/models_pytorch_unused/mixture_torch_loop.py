import numpy as np
import torch

def mixture_torch_loop(model,data,tol=1e-8,max_iter=100000,num_repl=1,init='no',LR=0.1):

    best_loglik = -1000000

    for repl in range(num_repl):
        if num_repl>1:
            print(['Initializing inner repl '+str(repl)])
        model.initialize(X=data,init=init,tol=tol)
        # the 'no'-option (default) is for ACG-lowrank, where some columns are randomly initialized and others prespecified
        
        optimizer = torch.optim.Adam(model.parameters(),lr=LR)
        # optimizer = torch.optim.SGD(model.parameters(),lr=LR)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,threshold=tol,threshold_mode='abs',min_lr=0.0001,patience=100)
        scheduler = None

        loglik = []

        for iter in range(max_iter):
            # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
            #     with record_function("model_inference"):
            #         epoch_nll = -model(data) #negative for nll
            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            epoch_nll = -model(data) #negative for nll

            if torch.isnan(-epoch_nll):
                raise ValueError("Nan reached")
            
            optimizer.zero_grad(set_to_none=True)
            epoch_nll.backward()
            optimizer.step()
            loglik.append(-epoch_nll.item())
            

            if iter>10:
                if scheduler is not None:
                    scheduler.step(epoch_nll)
                    if optimizer.param_groups[0]["lr"]<0.001:
                        break
                else:
                    if np.abs(loglik[-10]-loglik[-1])/np.abs(loglik[-1])<tol:
                        break
            if iter % 10 == 0:
                print(['Done with iteration '+str(iter)])
        
        if loglik[-1]>best_loglik:
            best_loglik = loglik[-1]
            loglik_final = loglik
            params_final = model.get_params()
            beta_final = model.posterior(X=data)            
            num_iter_final = iter


    return params_final,beta_final,loglik_final,num_iter_final
