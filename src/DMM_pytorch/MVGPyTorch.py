import torch
from src.DMM_pytorch.DMMPyTorch import DMMPyTorchBaseModel
import math

class MVG(DMMPyTorchBaseModel):
    def __init__(self, p:int, K:int=1, rank=None, HMM:bool=False, samples_per_sequence=None, params:dict=None,distribution_type='MVG_lowrank'):
        super().__init__()

        self.p = torch.tensor(p)
        self.half_p = torch.tensor(p/2)
        self.r = torch.tensor(rank)
        self.K = torch.tensor(K)
        self.HMM = HMM
        self.samples_per_sequence = samples_per_sequence
        self.distribution = distribution_type
        if self.distribution == 'MVG_diagonal':
            self.r = 1
        
        self.log_norm_constant = -self.half_p*torch.log(torch.tensor(2*math.pi))

        # initialize parameters
        if params is not None:
            self.unpack_params(params)
            
    def log_pdf(self,X):
        if self.distribution=='MVG_lowrank':
            Sigma = torch.eye(self.p) + self.M@torch.swapaxes(self.M,-2,-1)
            Sigma_inv = torch.linalg.inv(Sigma)
        elif self.distribution=='MVG_diagonal':
            Sigma = torch.diag_embed(self.M[:,:,0])
            Sigma_inv = torch.diag_embed(1/self.M[:,:,0])
        elif self.distribution=='MVG_scalar':
            Sigma = torch.zeros(self.K,self.p,self.p)
            Sigma_inv = torch.zeros(self.K,self.p,self.p)
            for k in range(self.K):
                Sigma[k] = self.sigma[k]*torch.eye(self.p)
                Sigma_inv[k] = 1/self.sigma[k]*torch.eye(self.p)

        log_pdf = self.log_norm_constant - 0.5 * torch.logdet(Sigma)[:,None] - 0.5 * torch.sum((X[None]-self.mu[:,None])@Sigma_inv*(X[None]-self.mu[:,None]),dim=-1)
        return log_pdf
