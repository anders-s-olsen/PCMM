import torch
from PCMM.PCMM_pytorch.PCMMPyTorchBaseModel import PCMMPyTorchBaseModel
import math
from PCMM.PCMM_pytorch.sqrtm import sqrtm

class Normal(PCMMPyTorchBaseModel):
    def __init__(self, p:int, rank:int, K:int=1, complex:bool=False, HMM:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = torch.tensor(p)
        self.r = torch.tensor(rank)
        self.K = torch.tensor(K)
        self.HMM = HMM
        self.samples_per_sequence = torch.tensor(samples_per_sequence)
        self.distribution = 'Normal_lowrank'
        self.complex = complex

        if complex:
            self.a = torch.tensor(1)
            self.c = self.p
            self.distribution = 'Complex_'+self.distribution
        else:
            self.a = torch.tensor(0.5)
            self.c = self.p/2
        
        self.log_norm_constant = -self.c*torch.log(1/self.a*torch.tensor(math.pi))
        self.norm_x = None

        # initialize parameters
        if params is not None:
            self.unpack_params(params)

    def log_pdf(self,X):

        gamma = torch.nn.functional.softplus(self.gamma)

        M_tilde = self.M*torch.sqrt(1/gamma[:,None,None])

        D = torch.swapaxes(torch.conj(M_tilde),-2,-1)@M_tilde+torch.eye(self.r)
        log_det_D = self.p*torch.log(gamma)+torch.logdet(D)
        
        D_sqrtinv = torch.zeros_like(D)
        for k in range(self.K):
            D_sqrtinv[k] = sqrtm(torch.linalg.inv(D[k]))

        # while Q_q^T Q_q != U_q^T L U_q, their determinants are the same
        if self.norm_x is None:
            self.norm_x = torch.linalg.norm(X,dim=1)**2
        
        XtM_tilde = torch.conj(X)[None,:,None,:]@M_tilde[:,None,:,:]

        v = 1/gamma[:,None]*(self.norm_x[None,:] - torch.linalg.norm(XtM_tilde@D_sqrtinv[:,None,:,:],dim=(-2,-1))**2)
        log_pdf = self.log_norm_constant - self.a*log_det_D.real[:,None] - self.a*v
        return log_pdf