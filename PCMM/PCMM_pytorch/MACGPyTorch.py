import torch
from PCMM.PCMM_pytorch.PCMMPyTorchBaseModel import PCMMPyTorchBaseModel
from PCMM.PCMM_pytorch.sqrtm import sqrtm

class MACG(PCMMPyTorchBaseModel):
    def __init__(self, p:int, q:int, rank:int, K:int=1, HMM:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = torch.tensor(p)
        self.q = torch.tensor(q)
        self.r = torch.tensor(rank)
        self.K = torch.tensor(K)
        self.HMM = HMM
        self.samples_per_sequence = torch.tensor(samples_per_sequence)
        self.distribution = 'MACG_lowrank'

        # initialize parameters
        if params is not None:
            self.unpack_params(params)

    def log_pdf(self,X):
        D = torch.swapaxes(self.M,-2,-1)@self.M+torch.eye(self.r)
        log_det_D = torch.logdet(D)
        D_sqrtinv = torch.zeros_like(D)
        for k in range(self.K):
            D_sqrtinv[k] = sqrtm(torch.linalg.inv(D[k]))

        XtM = torch.swapaxes(X,-2,-1)[None,:,:,:]@self.M[:,None,:,:]
        S2 = torch.linalg.svdvals(XtM@D_sqrtinv[:,None])
        v = torch.sum(torch.log(1/(S2**2)-1),dim=-1)+2*torch.sum(torch.log(S2),dim=-1)
        
        log_pdf = - (self.q/2)*log_det_D[:,None] - self.p/2*v
        return log_pdf