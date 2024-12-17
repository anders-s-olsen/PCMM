import torch
import torch.nn as nn
from PCMM.PCMMtorchBaseModel import PCMMtorchBaseModel
import math
from PCMM.sqrtm import sqrtm

class Watson(PCMMtorchBaseModel):
    def __init__(self, p:int, K:int=1, HMM:bool=False, complex:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = torch.tensor(p)
        self.K = torch.tensor(K)
        self.HMM = HMM
        self.samples_per_sequence = torch.tensor(samples_per_sequence)
        if complex:
            self.distribution = 'Complex_Watson'
            self.a = torch.tensor(1)
            self.c = torch.tensor(p)
        else:
            self.distribution = 'Watson'
            self.a = torch.tensor(0.5)
            self.c = torch.tensor(p/2)
        
        # precompute log-surface area of the unit hypersphere
        self.logSA_sphere = torch.lgamma(self.c) - torch.log(torch.tensor(2)) - self.c* torch.log(torch.tensor(math.pi))

        self.flag_normalized_input_data = False

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)

    def kummer_log(self,kappa, n=1e7,tol=1e-10):
        logkum = torch.zeros(self.K)
        logkum_old = torch.ones(self.K)
        foo = torch.zeros(self.K)
        for idx,k in enumerate(kappa):
            if k<0:
                a = self.c-self.a
            else:
                a = self.a
            j = 1
            while torch.abs(logkum[idx] - logkum_old[idx]) > tol and (j < n):
                logkum_old[idx] = logkum[idx]
                foo[idx] += torch.log((a + j - 1) / (j * (self.c + j - 1)) * torch.abs(k))
                logkum[idx] = torch.logsumexp(torch.stack((logkum[idx],foo[idx]),dim=0),dim=0)
                j += 1
        return logkum

    def log_norm_constant(self):
        return self.logSA_sphere - self.kummer_log(self.kappa)
    
    def log_pdf(self, X):
        if not self.flag_normalized_input_data:
            if not torch.allclose(torch.linalg.norm(X,axis=1),1):
                raise ValueError("For the Watson distribution, the input data vectors should be normalized to unit length.")
            else:
                self.flag_normalized_input_data = True
        mu_unit = nn.functional.normalize(self.mu, dim=1)
        logpdf = self.log_norm_constant()[:,None] + self.kappa[:,None]*(torch.abs(X@torch.conj(mu_unit).T)**2).T
        return logpdf #size (K,N)

class ACG(PCMMtorchBaseModel):
    def __init__(self, p:int, rank:int, K:int=1, HMM:bool=False, complex:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = torch.tensor(p)
        self.r = torch.tensor(rank)
        self.K = torch.tensor(K)
        self.HMM = HMM
        self.samples_per_sequence = torch.tensor(samples_per_sequence)
        self.distribution = 'ACG_lowrank'
        self.complex = complex

        if complex:
            self.a = torch.tensor(1)
            self.c = self.p
            self.distribution = 'Complex_'+self.distribution
        else:
            self.a = torch.tensor(0.5)
            self.c = self.p/2
        
        # precompute log-surface area of the unit hypersphere
        self.logSA_sphere = torch.lgamma(self.c) - torch.log(torch.tensor(2)) -self.c* torch.log(torch.tensor(math.pi))

        self.flag_normalized_input_data = False

        # initialize parameters
        if params is not None:
            self.unpack_params(params)
            
    def log_pdf(self,X):
        if not self.flag_normalized_input_data:
            if not torch.allclose(torch.linalg.norm(X,axis=1),1):
                raise ValueError("For the Watson distribution, the input data vectors should be normalized to unit length.")
            else:
                self.flag_normalized_input_data = True
        D = torch.eye(self.r) + torch.swapaxes(self.M,-2,-1).conj()@self.M
        XM = torch.conj(X[None,:,:])@self.M
        v = 1-torch.sum(XM@torch.linalg.inv(D)*torch.conj(XM),dim=-1) 
        log_pdf = self.logSA_sphere - self.a * torch.logdet(D).real[:,None] - self.c * torch.log(v.real)
        return log_pdf

class MACG(PCMMtorchBaseModel):
    def __init__(self, p:int, q:int, rank:int, K:int=1, HMM:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = torch.tensor(p)
        self.q = torch.tensor(q)
        self.r = torch.tensor(rank)
        self.K = torch.tensor(K)
        self.HMM = HMM
        self.samples_per_sequence = torch.tensor(samples_per_sequence)
        self.distribution = 'MACG_lowrank'

        self.flag_normalized_input_data = False

        # initialize parameters
        if params is not None:
            self.unpack_params(params)

    def log_pdf(self,X):
        if not self.flag_normalized_input_data:
            if torch.allclose(torch.linalg.norm(X[:,:,0],axis=1),1)!=1:
                raise ValueError("For the MACG distribution, the input data vectors should be normalized to unit length (and orthonormal, but this is not checked).")
            else:
                self.flag_normalized_input_data = True        
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
    
class SingularWishart(PCMMtorchBaseModel):
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
        self.log_norm_constant = self.q*(self.q-self.p)/2*torch.log(torch.tensor(math.pi))-self.p*self.q/2*torch.log(torch.tensor(2))-loggamma_q

        self.flag_normalized_input_data = False
        
        # initialize parameters
        if params is not None:
            self.unpack_params(params)

    def log_pdf(self,X):
        if not self.flag_normalized_input_data:
            X_weights = torch.linalg.norm(X,axis=1)**2
            if not torch.allclose(torch.sum(X_weights,axis=1),self.p):
                raise ValueError("In weighted grassmann clustering, the scale of the input data vectors should be equal to the square root of the eigenvalues. If the scale does not sum to the dimensionality, this error is thrown")
            else:
                self.flag_normalized_input_data = True

        gamma = torch.nn.functional.softplus(self.gamma)

        M_tilde = self.M*torch.sqrt(1/gamma[:,None,None])

        D = torch.swapaxes(M_tilde,-2,-1)@M_tilde+torch.eye(self.r)
        log_det_D = self.p*torch.log(gamma)+torch.logdet(D)
        
        D_sqrtinv = torch.zeros_like(D)
        for k in range(self.K):
            D_sqrtinv[k] = sqrtm(torch.linalg.inv(D[k]))

        # while Q_q^T Q_q != U_q^T L U_q, their determinants are the same
        if self.log_det_S11 is None:
            self.log_det_S11 = torch.logdet(torch.swapaxes(X[:,:self.q,:],-2,-1)@X[:,:self.q,:])
        
        QtM_tilde = torch.swapaxes(X,-2,-1)[None,:,:,:]@M_tilde[:,None,:,:]

        v = 1/gamma[:,None]*(self.p - torch.linalg.norm(QtM_tilde@D_sqrtinv[:,None],dim=(-2,-1))**2)
        log_pdf = self.log_norm_constant - (self.q/2)*log_det_D[:,None] + (self.q-self.p-1)/2*self.log_det_S11[None] - 1/2*v
        return log_pdf

class Normal(PCMMtorchBaseModel):
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