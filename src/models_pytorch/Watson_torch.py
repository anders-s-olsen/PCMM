import torch
import torch.nn as nn
import numpy as np
from src.models_python.diametrical_clustering_torch import diametrical_clustering_torch, diametrical_clustering_plusplus_torch
#from scipy.special import gamma, factorial

class Watson(nn.Module):
    """
    Logarithmic Multivariate Watson distribution class
    """

    def __init__(self, K,p,params=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.p = torch.tensor(p,device=self.device)
        self.c = torch.tensor(p/2,device=self.device)
        self.K = torch.tensor(K,device=self.device)
        self.a = torch.tensor(0.5,device=self.device)  # a = 1/2,  !constant
        self.logSA = torch.lgamma(self.c) - torch.log(torch.tensor(2,device=self.device)) -self.c* torch.log(torch.tensor(np.pi,device=self.device))

        if params is not None:
            self.mu = nn.Parameter(params['mu'])
            self.kappa = nn.Parameter(params['kappa']) # should be (K,1)
            if self.kappa.dim()==1:
                self.kappa = self.kappa[:,None]
            self.pi = nn.Parameter(params['pi'])

        self.LogSoftmax = nn.LogSoftmax(dim=0)
        self.Softplus = nn.Softplus()

        assert self.p != 1, 'Not properly implemented'

    def get_params(self):
        return {'mu': self.mu.data,'kappa':self.kappa.data,'pi':self.pi.data}
    
    def initialize(self,X=None,init=None):
        if init is None or init=='uniform' or init=='unif':
            self.mu = nn.Parameter(torch.nn.functional.normalize(torch.rand(size=(self.p,self.K)),dim=0))
        elif init == '++' or init == 'plusplus' or init == 'diametrical_clustering_plusplus':
            self.mu = nn.Parameter(diametrical_clustering_plusplus_torch(X=X,K=self.K))
        elif init == 'dc' or init == 'diametrical_clustering':
            self.mu = nn.Parameter(diametrical_clustering_torch(X=X,K=self.K,max_iter=100000,num_repl=5,init='++'))
            
        self.pi = nn.Parameter(torch.ones(self.K,device=self.device)/self.K)
        self.kappa = nn.Parameter(torch.ones((self.K,1)))

    # def log_kummer(self, a, b, kappa,num_eval=10000):

    #     n = torch.arange(num_eval,device=self.device)
    
    #     inner = torch.lgamma(a + n) + torch.lgamma(b) - torch.lgamma(a) - torch.lgamma(b + n) \
    #             + n * torch.log(kappa) - torch.lgamma(n + torch.tensor(1.,device=self.device))
    
    #     logkum = torch.logsumexp(inner, dim=0)
    
    #     return logkum

    def kummer_log(self,a, c, k, n=10000,tol=1e-10):
        logkum = torch.zeros((k.size(dim=0),1))
        logkum_old = torch.ones((k.size(dim=0),1))
        foo = torch.zeros((k.size(dim=0),1))
        j = 1
        while torch.any(torch.abs(logkum - logkum_old) > tol) and (j < n):
            logkum_old = logkum
            foo += torch.log((a + j - 1) / (j * (c + j - 1)) * k)
            logkum = torch.logsumexp(torch.stack((logkum,foo)),dim=0)
            j += 1
        return logkum    

    def log_norm_constant(self, kappa_pos):
        logC = self.logSA - self.kummer_log(self.a, self.c, kappa_pos)
        return logC

    def log_pdf(self, X):
        # Constraints
        kappa_positive = self.Softplus(self.kappa)  # Log softplus?
        mu_unit = nn.functional.normalize(self.mu, dim=0)  ##### Sufficent for backprop?

        if torch.any(torch.isnan(torch.log(kappa_positive))):
            raise ValueError('Too low kappa')

        norm_constant = self.log_norm_constant(kappa_positive)
        logpdf = norm_constant + kappa_positive * ((mu_unit.T @ X.T) ** 2)

        return logpdf
    
    def log_density(self,X):
        return self.log_pdf(X)+self.LogSoftmax(self.pi)[:,None]

    def log_likelihood(self, X):
        density = self.log_density(X)
        logsum_density = torch.logsumexp(density, dim=0)
        loglik = torch.sum(logsum_density)
        return loglik

    def forward(self, X):
        return self.log_likelihood(X)

    def __repr__(self):
        return 'Watson'
    
    def posterior(self,X):
        density = self.log_density(X)
        logsum_density = torch.logsumexp(density, dim=0)
        return torch.exp(density-logsum_density)


if __name__ == "__main__":
    # Test that the code works
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scipy
    mpl.use('Qt5Agg')

    W = Watson(p=2)
    print(W.log_kummer(torch.tensor(0.5),torch.tensor(100),torch.tensor(2)))
    print(W.log_kummer(torch.tensor(0.5),torch.tensor(100),torch.tensor(200)))
    print(W.log_kummer(torch.tensor(0.5),torch.tensor(100),torch.tensor(2000)))

    W_pdf = lambda phi: float(torch.exp(W(torch.tensor([np.cos(phi), np.sin(phi)], dtype=torch.float))))
    w_result = scipy.integrate.quad(W_pdf, 0., 2*np.pi)

    # phi = linspace(0, 2 * pi, 320);
    # x = [cos(phi);sin(phi)];
    #
    # _, inner = W.log_kummer(torch.tensor(0.5), torch.tensor(3/2), torch.tensor())

    phi = torch.arange(0, 2 * np.pi, 0.001)
    phi_arr = np.array(phi)
    x = torch.column_stack((torch.cos(phi), torch.sin(phi)))

    points = torch.exp(W(x))
    props = np.array(points.squeeze().detach())

    # props = props/np.max(props)

    ax = plt.axes(projection='3d')
    ax.plot(np.cos(phi_arr), np.sin(phi_arr), 0, 'gray')
    ax.scatter(np.cos(phi_arr), np.sin(phi_arr), props, c=props, cmap='viridis', linewidth=0.5)

    ax.view_init(30, 135)
    plt.show()