import numpy as np
import torch
from tqdm import tqdm
from TG import TorusGraphs
from src.DMM_EM.SingularWishartEM import SingularWishart

def mixture_torch_loop(X,noise,model):


    optimizer = torch.optim.Adam(model.parameters(),lr=0.1)
    objective = []


    for epoch in tqdm(range(100)):
            
        obj = -model.NCE_objective_function(X,noise) 


        if torch.isnan(-obj):
            raise ValueError("Nan reached")
        
        optimizer.zero_grad(set_to_none=True)
        obj.backward()
        optimizer.step()
        objective.append(-obj.item())
            
    return model,objective


if __name__=="__main__":
  
    #real data
    X = np.loadtxt('/dtu-compute/HCP_dFC/openneuro_sleep/phases_7/sub-01_mainlysleep_phases.txt')
    n,p = X.shape
    U_cos = np.zeros((n,p,2))
    L_cos = np.zeros((n,2))
    U_sin = np.zeros((n,p,2))
    L_sin = np.zeros((n,2))
    for i in range(n):
        A_cos = np.outer(np.cos(X[i]),np.cos(X[i])) + np.outer(np.sin(X[i]),np.sin(X[i]))
        L,U = np.linalg.eig(A_cos)
        order = np.argsort(L)[::-1]
        U_cos[i],L_cos[i] = U[:,order[:2]],L[order[:2]]
        A_sin = np.sin(X[i])[:,None]*np.cos(X[i])[None,:] - np.cos(X[i])[:,None]*np.sin(X[i])[None,:]
        A_sin = np.outer(np.sin(X[i]),np.cos(X[i])) - np.outer(np.cos(X[i]),np.sin(X[i]))
        L,U = np.linalg.eig(A_sin)
        order = np.argsort(L)[::-1]
        U_sin[i],L_sin[i] = U[:,order[:2]],L[order[:2]]

    # fit unimodal Singular Wishart distribution
    SW = SingularWishart(p=p,q=2,K=1)
    Lambda_cos = SW.M_step_single_component(U_cos,L_cos,beta=np.ones(n))
    Lambda_sin = SW.M_step_single_component(U_sin,L_sin,beta=np.ones(n))

    import matplotlib.pyplot as plt
    # plot the two inferred matrices with colorbars
    plt.figure()
    plt.subplot(121)
    plt.imshow(Lambda_cos)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(Lambda_sin)
    plt.colorbar()
    plt.savefig('tmp.png')

    X = torch.tensor([0,0,0])+torch.rand(N,3)*0.1


    noise = torch.rand(N,p)*2*torch.tensor(np.pi)


    model = TorusGraphs(p=X.shape[1],K=1)
    model,objective = mixture_torch_loop(X,noise,model)


    theta = model.theta


    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(objective)
    plt.savefig('tmp.png')