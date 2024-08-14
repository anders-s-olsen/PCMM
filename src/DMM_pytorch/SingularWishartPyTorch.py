import torch
from src.DMM_pytorch.DMMPyTorch import DMMPyTorchBaseModel
import math
from src.DMM_pytorch.sqrtm import sqrtm

class SingularWishart(DMMPyTorchBaseModel):
    def __init__(self, p:int, q:int, rank:int, K:int=1, HMM:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = torch.tensor(p)
        self.q = torch.tensor(q)
        self.r = torch.tensor(rank)
        self.K = torch.tensor(K)
        self.HMM = HMM
        self.samples_per_sequence = torch.tensor(samples_per_sequence)
        self.distribution = 'SingularWishart_lowrank'
        self.log_det_S11 = None
        
        loggamma_q = (self.q*(self.q-1)/4)*torch.log(torch.tensor(math.pi))+torch.sum(torch.lgamma(self.q-torch.arange(self.q)/2))
        self.log_norm_constant = self.q*(self.q-self.p)*torch.log(torch.tensor(math.pi))-self.p*self.q/2*torch.log(torch.tensor(2))-loggamma_q
        
        # initialize parameters
        if params is not None:
            self.unpack_params(params)

    def log_pdf(self,Q):

        gamma = torch.nn.functional.softplus(self.gamma)

        M_tilde = self.M*torch.sqrt(1/gamma[:,None,None])

        D = torch.swapaxes(M_tilde,-2,-1)@M_tilde+torch.eye(self.r)
        log_det_D = self.p*torch.log(gamma)+torch.logdet(D)
        
        D_sqrtinv = torch.zeros_like(D)
        for k in range(self.K):
            D_sqrtinv[k] = sqrtm(torch.linalg.inv(D[k]))

        # while Q_q^T Q_q != U_q^T L U_q, their determinants are the same
        if self.log_det_S11 is None:
            self.log_det_S11 = torch.logdet(torch.swapaxes(Q[:,:self.q,:],-2,-1)@Q[:,:self.q,:])
        
        QtM_tilde = torch.swapaxes(Q,-2,-1)[None,:,:,:]@M_tilde[:,None,:,:]

        v = 1/gamma[:,None]*(self.p - torch.linalg.norm(QtM_tilde@D_sqrtinv[:,None],dim=(-2,-1))**2)
        log_pdf = self.log_norm_constant - (self.q/2)*log_det_D[:,None] + (self.q-self.p-1)/2*self.log_det_S11[None] - 1/2*v
        return log_pdf
