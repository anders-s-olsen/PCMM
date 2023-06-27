import torch
import torch.nn as nn


class TorchMixtureModel(nn.Module):
    """
    Mixture model class
    """
    def __init__(self, dist, K: int, p: int,D = None,regu=0,init=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.K = K
        self.p = p
        self.D = D #for lowrank ACG
        self.dist = dist
        if init is None:
            self.pi = nn.Parameter(torch.rand(self.K,device=self.device))
            self.mix_components = nn.ModuleList([self.dist(self.p,self.D) for _ in range(self.K)])
        else:
            self.pi = nn.Parameter(init['pi'])
            self.mix_components = nn.ModuleList([self.dist(self.p,self.D,init['comp'][k]) for k in range(self.K)])
        self.LogSoftMax = nn.LogSoftmax(dim=0)
        self.softplus = nn.Softplus()
        

    @torch.no_grad()
    def get_model_param(self):
        un_norm_pi = self.pi.data
        mixture_param_dict = {'un_norm_pi': un_norm_pi}
        for comp_id, comp_param in enumerate(self.mix_components):
            mixture_param_dict[f'mix_comp_{comp_id}'] = comp_param.get_params()
        return mixture_param_dict

    def log_likelihood_mixture(self, X):
        inner_pi = self.LogSoftMax(self.softplus(self.pi))[:, None]
        inner_pdf = torch.stack([K_comp_pdf(X) for K_comp_pdf in self.mix_components]) #one component at a time but all X is input

        inner = inner_pi + inner_pdf

        loglikelihood_x_i = torch.logsumexp(inner, dim=0)  # Log likelihood over a sample of p-dimensional vectors

        logLikelihood = torch.sum(loglikelihood_x_i)
        return logLikelihood

    def forward(self, X):
        return self.log_likelihood_mixture(X)

    def posterior(self,X):
        inner_pi = self.LogSoftMax(self.softplus(self.pi))[:, None]
        inner_pdf = torch.stack([K_comp_pdf(X) for K_comp_pdf in self.mix_components])

        inner = inner_pi + inner_pdf
        loglikelihood_x_i = torch.logsumexp(inner, dim=0)
        return torch.exp(inner-loglikelihood_x_i),
        
if __name__ == "__main__":
    from Watson_torch import Watson
    from ACG_lowrank_torch import ACG
    # from MACG_lowrank_torch import MACG
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    K = torch.tensor(2)
    
    p = torch.tensor(3)
    
    model = TorchMixtureModel(Watson,K=K,p=p,D=3)

    data = torch.tensor(np.loadtxt('data/synthetic/synth_data_ACG.csv',delimiter=','))
    optimizer = torch.optim.Adam(model.parameters(),lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,threshold=1e-8,threshold_mode='abs',min_lr=0.0001,patience=100)
   
    num_epoch = 10000
    epoch_nll_collector = []

    for epoch in tqdm(range(num_epoch)):
        

        epoch_nll = -model(data) #negative for nll

        optimizer.zero_grad(set_to_none=True)
        epoch_nll.backward()
        optimizer.step()
        epoch_nll_collector.append(epoch_nll.detach())

        if epoch>100:
            if scheduler is not None:
                scheduler.step(epoch_nll)
                if optimizer.param_groups[0]["lr"]<0.001:
                    break

    a = model.get_model_param()
    L_tri_inv = a['mix_comp_1']['L_tri_inv']
    L = torch.linalg.inv(L_tri_inv@L_tri_inv.T)
    p*L/torch.trace(L)

    stop=7
