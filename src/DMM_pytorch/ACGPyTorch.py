import torch
from src.DMM_pytorch.DMMPyTorch import DMMPyTorchBaseModel
import math

class ACG(DMMPyTorchBaseModel):
    def __init__(self, p:int, K:int=1, rank=None, HMM:bool=False, samples_per_sequence=None, params:dict=None):
        super().__init__()

        self.p = torch.tensor(p)
        self.half_p = torch.tensor(p/2)
        self.r = torch.tensor(rank)
        self.K = torch.tensor(K)
        self.HMM = HMM
        self.samples_per_sequence = samples_per_sequence
        self.distribution = 'ACG'
        
        # precompute log-surface area of the unit hypersphere
        self.logSA_sphere = torch.lgamma(self.half_p) - torch.log(torch.tensor(2)) -self.half_p* torch.log(torch.tensor(math.pi))

        # initialize parameters
        if params is not None:
            self.unpack_params(params)
            
    def log_pdf(self,X,L=None):
        D = torch.eye(self.r) + torch.swapaxes(self.M,-2,-1)@self.M
        XM = X[None,:,:]@self.M
        v = 1-torch.sum(XM@torch.linalg.inv(D)*XM,dim=-1) #check
        log_pdf = self.logSA_sphere - 0.5 * torch.logdet(D)[:,None] - self.half_p * torch.log(v)
        return log_pdf
    