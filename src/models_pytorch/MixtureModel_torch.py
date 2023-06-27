import torch
import torch.nn as nn


class TorchMixtureModel(nn.Module):
    """
    Mixture model class
    """
    def __init__(self, distribution_object, K: int, dist_dim=90,D = None,regu=0,init=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.regu = regu # for regularized cholesky decomposition
        self.D = D #for lowrank ACG
        self.distribution, self.K, self.p = distribution_object, K, dist_dim
        if init is None:
            self.pi = nn.Parameter(torch.rand(self.K,device=self.device))
            self.mix_components = nn.ModuleList([self.distribution(self.p,self.D) for _ in range(self.K)])
        else:
            self.pi = nn.Parameter(init['pi'])
            self.mix_components = nn.ModuleList([self.distribution(self.p,self.D,init['comp'][k]) for k in range(self.K)])
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
        #inner = inner_pdf
        # print(torch.exp(inner))

        loglikelihood_x_i = torch.logsumexp(inner, dim=0)  # Log likelihood over a sample of p-dimensional vectors
        # print(loglikelihood_x_i)

        logLikelihood = torch.sum(loglikelihood_x_i)
        # print(logLikeLihood)
        return logLikelihood

    def forward(self, X):
        return self.log_likelihood_mixture(X)

    def posterior(self,X):
        inner_pi = self.LogSoftMax(self.softplus(self.pi))[:, None]
        inner_pdf = torch.stack([K_comp_pdf(X) for K_comp_pdf in self.mix_components])

        inner = inner_pi + inner_pdf
        loglikelihood_x_i = torch.logsumexp(inner, dim=0)

        return torch.exp(inner-loglikelihood_x_i)
        
if __name__ == "__main__":
    # test that the code works
    from Watson_torch import Watson
    from AngularCentralGauss_chol import AngularCentralGaussian

    torch.set_printoptions(precision=4)
    MW = TorchMixtureModel(Watson, K=2, dist_dim=3)

    data = torch.rand(6, 3)
    out = MW(data)
    print(data)
    print(MW(data)) #likelihood of data
