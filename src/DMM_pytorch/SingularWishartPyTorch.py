import torch
from src.DMM_pytorch.DMMPyTorch import DMMPyTorchBaseModel
import math
class SingularWishart(DMMPyTorchBaseModel):
    def __init__(self, p:int, q:int, K:int=1, rank=None, HMM:bool=False, samples_per_sequence=None, params:dict=None):
        super().__init__()

        self.p = torch.tensor(p)
        self.q = torch.tensor(q)
        self.half_p = torch.tensor(p/2)
        self.r = torch.tensor(rank)
        self.K = torch.tensor(K)
        self.HMM = HMM
        self.samples_per_sequence = samples_per_sequence
        self.distribution = 'SingularWishart'
        
        self.log_norm_constant = -self.p*self.q/2*torch.log(2*math.pi)

        # initialize parameters
        if params is not None:
            self.unpack_params(params)

    def log_pdf_lowrank(self, X,L):
        D = torch.eye(self.r) + torch.swapaxes(self.M,-2,-1)@self.M 
        XtM = torch.swapaxes(X,-2,-1)[None,:,:,:]@self.M[:,None,:,:]

        v = torch.sum(L,dim=(-2,-1))-torch.trace(XtM@torch.linalg.inv(D)[:,None]@torch.swapaxes(XtM,-2,-1)@L)

        log_pdf = self.log_norm_constant - (self.q/2)*torch.logdet(D) - 1/2*v
        return log_pdf
