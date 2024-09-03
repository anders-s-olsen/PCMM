import torch
from src.DMM_pytorch.DMMPyTorch import DMMPyTorchBaseModel
import math

class ACG(DMMPyTorchBaseModel):
    def __init__(self, p:int, rank:int, K:int=1, HMM:bool=False, complex:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = torch.tensor(p)
        self.r = torch.tensor(rank)
        self.K = torch.tensor(K)
        self.HMM = HMM
        self.samples_per_sequence = torch.tensor(samples_per_sequence)
        self.distribution = 'ACG_lowrank'

        if complex:
            self.a = torch.tensor(1)
            self.c = self.p
            self.distribution = 'Complex_'+self.distribution
        else:
            self.a = torch.tensor(0.5)
            self.c = self.p/2
        
        # precompute log-surface area of the unit hypersphere
        self.logSA_sphere = torch.lgamma(self.c) - torch.log(torch.tensor(2)) -self.c* torch.log(torch.tensor(math.pi))

        # initialize parameters
        if params is not None:
            self.unpack_params(params)
            
    def log_pdf(self,X):
        D = torch.eye(self.r) + torch.swapaxes(self.M,-2,-1).conj()@self.M
        XM = torch.conj(X[None,:,:])@self.M
        v = 1-torch.sum(XM@torch.linalg.inv(D)*torch.conj(XM),dim=-1) 
        log_pdf = self.logSA_sphere - self.a * torch.logdet(D).real[:,None] - self.c * torch.log(v.real)
        return log_pdf