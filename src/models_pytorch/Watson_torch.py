import torch
import torch.nn as nn
import numpy as np
from src.load_HCP_data import initialize_pi_mu_M

class Watson(nn.Module):
    """
    Logarithmic Multivariate Watson distribution class
    """

    def __init__(self, K,p,params=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.p = torch.tensor(p,device=self.device)
        self.half_p = torch.tensor(p/2,device=self.device)
        self.K = torch.tensor(K,device=self.device)
        self.a = torch.tensor(0.5,device=self.device)  # a = 1/2,  !constant
        self.logSA = torch.lgamma(self.half_p) - torch.log(torch.tensor(2,device=self.device)) -self.half_p* torch.log(torch.tensor(np.pi,device=self.device))

        if params is not None:
            if torch.is_tensor(params['pi']):
                self.mu = nn.Parameter(params['mu'])
                self.kappa = nn.Parameter(params['kappa']) # should be (K,1)
                self.pi = nn.Parameter(params['pi'])
            else:
                self.mu = nn.Parameter(torch.tensor(params['mu']))
                self.kappa = nn.Parameter(torch.tensor(params['kappa'])) # should be (K,1)
                self.pi = nn.Parameter(torch.tensor(params['pi']))

        self.LogSoftmax = nn.LogSoftmax(dim=0)
        # self.Softplus = nn.Softplus(beta=20, threshold=1)

        assert self.p != 1, 'Not properly implemented'

    def get_params(self):
        return {'mu': self.mu.data,'kappa':self.kappa.data,'pi':self.pi.data}
    
    def initialize(self,X=None,init=None,tol=None):
        pi,mu,_ = initialize_pi_mu_M(init=init,K=self.K,p=self.p,X=X) 
        self.pi = nn.Parameter(torch.tensor(pi))
        self.mu = nn.Parameter(torch.tensor(mu))
        self.kappa = nn.Parameter(torch.ones(self.K,device=self.device)) 

    def kummer_log(self,a, c, kappa, n=1000000,tol=1e-10):
        if kappa.ndim==0:
            kappa_size = 1
        else:
            kappa_size = kappa.size(dim=0)
        if torch.any(kappa<0):
            logkum = torch.zeros(kappa_size)
            for idx,k in enumerate(kappa):
                if k<0:
                    logkum[idx] = self.kummer_log_neg(a=a,c=c,kappa=torch.tensor(k),n=n,tol=tol)
                else:
                    logkum[idx] = self.kummer_log(a=a,c=c,kappa=torch.tensor(k),n=n,tol=tol)
            return logkum

        logkum = torch.zeros(kappa_size)
        logkum_old = torch.ones(kappa_size)
        foo = torch.zeros(kappa_size)
        j = 1
        while torch.any(torch.abs(logkum - logkum_old) > tol) and (j < n):
            logkum_old = logkum
            foo += torch.log((a + j - 1) / (j * (c + j - 1)) * kappa)
            logkum = torch.logsumexp(torch.stack((logkum,foo),dim=0),dim=0)
            j += 1
        return logkum     
    def kummer_log_neg(self,a, c, kappa, n=1000000,tol=1e-10):
        logkum = torch.zeros(1)
        logkum_old = torch.ones(1)
        foo = torch.zeros(1)
        j = 1
        a = c-a
        while torch.any(torch.abs(logkum - logkum_old) > tol) and (j < n):
            logkum_old = logkum
            foo += torch.log((a + j - 1) / (j * (c + j - 1)) * torch.abs(kappa))
            logkum = torch.logsumexp(torch.stack((logkum,foo),dim=0),dim=0)
            j += 1
        return logkum+kappa    

    def log_norm_constant(self, kappa):
        logC = self.logSA - self.kummer_log(self.a, self.half_p, kappa)[:,None]
        return logC

    def log_pdf(self, X):
        # Constraints
        # kappa = self.Softplus(self.kappa)  # Log softplus?
        kappa = self.kappa
        mu_unit = nn.functional.normalize(self.mu, dim=0)  ##### Sufficent for backprop?

        # if torch.any(torch.isinf(torch.log(kappa))):
        #     raise ValueError('Too low kappa')

        norm_constant = self.log_norm_constant(kappa)
        logpdf = norm_constant + kappa[:,None] * ((mu_unit.T @ X.T) ** 2)

        return logpdf
    
    def log_density(self,X):
        return self.log_pdf(X)+self.LogSoftmax(self.pi)[:,None]

    def log_likelihood(self, X):
        density = self.log_density(X)
        logsum_density = torch.logsumexp(density, dim=0)
        loglik = torch.sum(logsum_density)
        if torch.isnan(loglik):
            raise ValueError('nan reached')
        return loglik
    
    def test_log_likelihood(self, X): #without constraints (assumed mu normalized and pi sum to one)
        norm_constant = self.log_norm_constant(self.kappa)
        logpdf = norm_constant + self.kappa[:,None] * ((self.mu.T @ X.T) ** 2)
        density = logpdf+torch.log(self.pi[:,None])
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

