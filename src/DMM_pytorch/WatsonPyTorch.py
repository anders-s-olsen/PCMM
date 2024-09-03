import torch
import torch.nn as nn
from src.DMM_pytorch.DMMPyTorch import DMMPyTorchBaseModel
import math

class Watson(DMMPyTorchBaseModel):
    def __init__(self, p:int, K:int=1, HMM:bool=False, complex:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = torch.tensor(p)
        self.K = torch.tensor(K)
        self.HMM = HMM
        self.samples_per_sequence = torch.tensor(samples_per_sequence)
        if complex:
            self.distribution = 'Complex_Watson'
            self.a = torch.tensor(1)
            self.c = torch.tensor(p)
        else:
            self.distribution = 'Watson'
            self.a = torch.tensor(0.5)
            self.c = torch.tensor(p/2)
        
        # precompute log-surface area of the unit hypersphere
        self.logSA_sphere = torch.lgamma(self.c) - torch.log(torch.tensor(2)) - self.c* torch.log(torch.tensor(math.pi))

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)

    def kummer_log(self,kappa, n=1e7,tol=1e-10):
        logkum = torch.zeros(self.K)
        logkum_old = torch.ones(self.K)
        foo = torch.zeros(self.K)
        for idx,k in enumerate(kappa):
            if k<0:
                a = self.c-self.a
            else:
                a = self.a
            j = 1
            while torch.abs(logkum[idx] - logkum_old[idx]) > tol and (j < n):
                logkum_old[idx] = logkum[idx]
                foo[idx] += torch.log((a + j - 1) / (j * (self.c + j - 1)) * torch.abs(k))
                logkum[idx] = torch.logsumexp(torch.stack((logkum[idx],foo[idx]),dim=0),dim=0)
                j += 1
        return logkum

    def log_norm_constant(self):
        return self.logSA_sphere - self.kummer_log(self.kappa)
    
    def log_pdf(self, X):
        mu_unit = nn.functional.normalize(self.mu, dim=0)
        logpdf = self.log_norm_constant()[:,None] + self.kappa[:,None]*(torch.abs(X@torch.conj(mu_unit))**2).T
        return logpdf #size (K,N)
        