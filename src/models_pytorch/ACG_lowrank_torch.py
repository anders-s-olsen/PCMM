import numpy as np
import torch
import torch.nn as nn
from src.helper_functions2 import initialize_pi_mu_M

class ACG(nn.Module):
    """
    Angular-Central-Gaussian spherical distribution:

    "Tyler 1987 - Statistical analysis for the angular central Gaussian distribution on the sphere"
    """

    def __init__(self, K: int,p: int,rank=None,params=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.K = K
        self.p = torch.tensor(p) #dimensionality
        self.r = rank
        self.half_p = torch.tensor(p / 2)
        self.logSA = torch.lgamma(self.half_p) - torch.log(torch.tensor(2)) -self.half_p* torch.log(torch.tensor(np.pi))

        self.LogSoftmax = nn.LogSoftmax(dim=0)
        
        if params is not None: # for evaluating likelihood with already-learned parameters
            if torch.is_tensor(params['pi']):
                self.pi = nn.Parameter(params['pi'])
            else:
                self.pi = nn.Parameter(torch.tensor(params['pi']))

            M_init = params['M']
            if M_init.dim()!=3 or M_init.shape[2]!=self.r: # add extra columns
                if M_init.dim()==2:
                    num_missing = self.r-1
                else:
                    num_missing = self.r-M_init.shape[2]

                M_extra = torch.randn(self.K,self.p,num_missing)
                self.M = nn.Parameter(torch.cat([M_init,M_extra],dim=2))
            else: 
                self.M = M_init
            

    def get_params(self):
        return {'M':self.M.data,'pi':self.pi.data} #should be normalized: L=MM^T+I and then L = p*L/trace(L)

    def initialize(self,X=None,init=None,tol=None):
        if init == 'no':
            return
        pi,_,M = initialize_pi_mu_M(init=init,K=self.K,p=self.p,X=X,tol=tol,r=self.r,initM=True)
        pi = nn.Parameter(torch.tensor(pi))
        M = nn.Parameter(torch.tensor(M))
        
    def log_pdf(self,X):
        D = torch.eye(self.r) + torch.swapaxes(self.M,-2,-1)@self.M
        log_det_D = torch.logdet(D)
        XM = X[None,:,:]@self.M
        pdf = 1-torch.sum(XM@torch.linalg.inv(D)*XM,dim=2) #check

        # minus log_det_L instead of + log_det_A_inv
        log_acg_pdf = self.logSA - 0.5 * log_det_D[:,None] - self.half_p * torch.log(pdf)
        return log_acg_pdf
    
    def log_density(self,X):
        return self.log_pdf(X)+self.LogSoftmax(self.pi)[:,None]

    def log_likelihood(self, X):
        density = self.log_density(X)
        logsum_density = torch.logsumexp(density, dim=0)
        loglik = torch.sum(logsum_density)
        return loglik
    
    def test_log_likelihood(self, X): #without constraints (assume pi sum to one)
        density = self.log_pdf(X)+torch.log(self.pi)[:,None]
        logsum_density = torch.logsumexp(density, dim=0)
        loglik = torch.sum(logsum_density)
        return loglik    

    def forward(self, X):
        return self.log_likelihood(X)
    
    def posterior(self,X):
        density = self.log_density(X)
        logsum_density = torch.logsumexp(density, dim=0)
        return torch.exp(density-logsum_density)
    
    def __repr__(self):
        return 'ACG'

