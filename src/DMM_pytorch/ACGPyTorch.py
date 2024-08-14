import torch
from src.DMM_pytorch.DMMPyTorch import DMMPyTorchBaseModel
import math

class ACG(DMMPyTorchBaseModel):
    def __init__(self, p:int, rank:int, K:int=1, HMM:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = torch.tensor(p)
        self.r = torch.tensor(rank)
        self.K = torch.tensor(K)
        self.HMM = HMM
        self.samples_per_sequence = torch.tensor(samples_per_sequence)
        self.distribution = 'ACG_lowrank'
        
        # precompute log-surface area of the unit hypersphere
        self.logSA_sphere = torch.lgamma(self.p/2) - torch.log(torch.tensor(2)) -self.p/2* torch.log(torch.tensor(math.pi))

        # initialize parameters
        if params is not None:
            self.unpack_params(params)
            
    def log_pdf(self,X):
        D = torch.eye(self.r) + torch.swapaxes(self.M,-2,-1)@self.M
        XM = X[None,:,:]@self.M
        v = 1-torch.sum(XM@torch.linalg.inv(D)*XM,dim=-1) 
        log_pdf = self.logSA_sphere - 0.5 * torch.logdet(D)[:,None] - self.p/2 * torch.log(v)
        return log_pdf
