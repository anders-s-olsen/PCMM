import torch
from src.DMM_pytorch.DMMPyTorch import DMMPyTorchBaseModel
import math
# from src.DMM_pytorch.sqrtm import sqrtm

class SingularWishart(DMMPyTorchBaseModel):
    def __init__(self, p:int, q:int, rank:int, K:int=1, HMM:bool=False, samples_per_sequence=None, params:dict=None):
        super().__init__()

        self.p = torch.tensor(p)
        self.q = torch.tensor(q)
        self.half_p = torch.tensor(p/2)
        self.r = torch.tensor(rank)
        self.K = torch.tensor(K)
        self.HMM = HMM
        self.samples_per_sequence = samples_per_sequence
        self.distribution = 'SingularWishart_lowrank'
        self.log_det_S11 = None
        
        # self.log_norm_constant = -self.p*self.q/2*torch.log(torch.tensor(2*math.pi))

        loggamma_k = (self.q*(self.q-1)/4)*torch.log(torch.tensor(math.pi))+torch.sum(torch.lgamma(self.q-torch.arange(self.q)/2))
        self.log_norm_constant = self.q**2/2*torch.log(torch.tensor(math.pi))+loggamma_k-self.q*self.half_p*torch.log(2*torch.tensor(math.pi))

        # initialize parameters
        if params is not None:
            self.unpack_params(params)

    # def log_pdf(self, Q):
    #     D = torch.eye(self.r) + torch.swapaxes(self.M,-2,-1)@self.M 
    #     QtM = torch.swapaxes(Q,-2,-1)[:,None,:,:]@self.M[None,:,:,:]

    #     D_sqrtinv = torch.zeros_like(D)
    #     for k in range(self.K):
    #         D_sqrtinv[k] = sqrtm(torch.linalg.inv(D[k]))

    #     v = self.p - torch.linalg.norm(QtM@D_sqrtinv,dim=(-2,-1))**2

    #     log_pdf = self.log_norm_constant - (self.q/2)*torch.logdet(D)[:,None] - 1/2*v.T
    #     return log_pdf
    def log_pdf(self,Q):
        
        QtM = torch.swapaxes(Q,-2,-1)[None,:,:,:]@self.M[:,None,:,:]

        _,S1,V1t = torch.svd(self.M)
        D_sqrtinv = (torch.swapaxes(V1t,-2,-1)*torch.sqrt(1/(1+S1**2))[:,None])@V1t
        log_det_D = torch.sum(torch.log(1/S1**2+1),dim=-1)+2*torch.sum(torch.log(S1),dim=-1)

        # while Q_q^T Q_q != U_q^T L U_q, their determinants are the same
        if self.log_det_S11 is None:
            self.log_det_S11 = torch.logdet(torch.swapaxes(Q[:,:self.q,:],-2,-1)@Q[:,:self.q,:])

        v = self.p - torch.linalg.norm(QtM@D_sqrtinv[:,None],dim=(-2,-1))**2
        log_pdf = self.log_norm_constant - (self.q/2)*log_det_D[:,None] + (self.q-self.p-1)/2*self.log_det_S11 - 1/2*v
        return log_pdf
    