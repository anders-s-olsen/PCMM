import torch
from src.DMM_pytorch.DMMPyTorch import DMMPyTorchBaseModel
import math
class MACG(DMMPyTorchBaseModel):
    def __init__(self, p:int, q:int, K:int=1, rank=None, HMM:bool=False, samples_per_sequence=None, params:dict=None):
        super().__init__()

        self.p = torch.tensor(p)
        self.q = torch.tensor(q)
        self.half_p = torch.tensor(p/2)
        self.r = torch.tensor(rank)
        self.K = torch.tensor(K)
        self.HMM = HMM
        self.samples_per_sequence = samples_per_sequence
        self.distribution = 'MACG'
        
        # precompute log-surface area of the Stiefel manifold
        loggamma_k = (self.q*(self.q-1)/4)*torch.log(torch.tensor(math.pi))+torch.sum(torch.lgamma(self.half_p-torch.arange(self.q)/2))
        self.logSA_stiefel = loggamma_k - self.q*torch.log(torch.tensor(2)) - self.q*self.half_p*torch.log(torch.tensor(math.pi))

        # initialize parameters
        if params is not None:
            self.unpack_params(params)

    def log_pdf(self,X,L=None):
        D = torch.eye(self.r) + torch.swapaxes(self.M,-2,-1)@self.M
        log_det_D = torch.logdet(D)
        XtM = torch.swapaxes(X,-2,-1)[None,:,:,:]@self.M[:,None,:,:]
        v = torch.logdet(D[:,None]-torch.swapaxes(XtM,-2,-1)@XtM)-log_det_D[:,None]

        log_pdf = self.logSA_stiefel - (self.q/2)*log_det_D[:,None] - self.half_p*v
        return log_pdf
    