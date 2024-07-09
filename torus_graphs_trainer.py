import numpy as np
import torch
from tqdm import tqdm
from TorusGraphsNCE import TorusGraphs


def mixture_torch_loop(X,noise,model):


   optimizer = torch.optim.Adam(model.parameters(),lr=0.1)
   objective = []


   for epoch in tqdm(range(1000)):
       obj = -model.NCE_objective_function(X,noise) 


       if torch.isnan(-obj):
           raise ValueError("Nan reached")
       
       optimizer.zero_grad(set_to_none=True)
       obj.backward()
       optimizer.step()
       objective.append(-obj.item())
       
       return model,objective


if __name__=="__main__":
  N=1000
  p=3


  X = torch.tensor([0,0,0])+torch.rand(N,3)*0.1


  noise = torch.rand(N,p)*2*torch.tensor(np.pi)


  model = TorusGraphs(p=X.shape[1],K=1)
  model,objective = mixture_torch_loop(X,noise,model)


  theta = model.theta


  import matplotlib.pyplot as plt
  plt.figure()
  plt.plot(objective)
  plt.savefig('tmp.png')