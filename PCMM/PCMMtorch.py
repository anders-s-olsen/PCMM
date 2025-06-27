import torch
import torch.nn as nn
from PCMM.PCMMtorchBaseModel import PCMMtorchBaseModel
import math

class Watson(PCMMtorchBaseModel):
    """
    Watson distribution on the (complex) projective hyperplane.
    The Watson distirbution is parameterized by a concentration parameter kappa and a mean direction mu.
    Oppositely the other distributions in this file, it is NOT parameterized by a covariance matrix.
    The Watson distribution will fail if the input data vectors are not normalized to unit length.
    Args:
        p (int): dimensionality of the data vectors
        K (int): number of clusters
        HMM (bool): whether to use the HMM variant of the model (default: False)
        complex (bool): whether to use the complex Watson distribution (default: False)
        samples_per_sequence (int): number of samples per sequence to be used by HMM (default: 0, meaning one long sequence)
        params (dict): dictionary containing the parameters of the model, if available (default: None
    """
    def __init__(self, p:int, K:int=1, HMM:bool=False, complex:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = p
        self.K = K
        self.HMM = HMM
        if samples_per_sequence is None:
            samples_per_sequence = 0
        self.samples_per_sequence = torch.as_tensor(samples_per_sequence)
        if complex:
            self.distribution = 'Complex_Watson'
            self.a = torch.as_tensor(1)
            self.c = torch.as_tensor(p)
        else:
            self.distribution = 'Watson'
            self.a = torch.as_tensor(0.5)
            self.c = torch.as_tensor(p/2)
        
        # precompute log-surface area of the unit hypersphere
        self.logSA_sphere = torch.lgamma(self.c) - torch.log(torch.as_tensor(2)) - self.c* torch.log(torch.as_tensor(math.pi))

        self.flag_normalized_input_data = False

        # initialize parameters
        if params is not None:
            self.unpack_params(params)

    def kummer_log(self,kappa, n=1e7,tol=1e-10):
        """ 
        Logarithm of the Kummer function for each kappa value.
        Args:
            kappa (torch.Tensor): A tensor of shape (K,) containing the kappa values.
            n (int): The maximum number of terms to compute in the series.
            tol (float): The tolerance for convergence.
        Returns:
            torch.Tensor: A tensor of shape (K,) containing the logarithm of the Kummer function for each kappa value.
        """
        logkum = torch.zeros(self.K)
        logkum_old = torch.ones(self.K)
        tmp = torch.zeros(self.K)
        for idx,k in enumerate(kappa):
            if k<0:
                a = self.c-self.a
            else:
                a = self.a
            j = 1
            # I modified this from somewhere but I cannot remember where :(
            while torch.abs(logkum[idx] - logkum_old[idx]) > tol and (j < n):
                logkum_old[idx] = logkum[idx]
                tmp[idx] += torch.log((a + j - 1) / (j * (self.c + j - 1)) * torch.abs(k))
                logkum[idx] = torch.logsumexp(torch.stack((logkum[idx],tmp[idx]),dim=0),dim=0)
                j += 1
        return logkum

    def log_norm_constant(self):
        return self.logSA_sphere - self.kummer_log(self.kappa)
    
    def log_pdf(self, X, recompute_statics=False):
        # check for normalized input data
        if not self.flag_normalized_input_data:
            if not torch.allclose(torch.linalg.norm(X,axis=1),torch.as_tensor(1.,dtype=torch.double)):
                raise ValueError("For the Watson distribution, the input data vectors should be normalized to unit length.")
            else:
                self.flag_normalized_input_data = True

        # reparameterize mu to be unit norm
        mu_unit = nn.functional.normalize(self.mu, dim=1)

        # compute logpdf of Watson distribution for each component
        logpdf = self.log_norm_constant().unsqueeze(-1) + self.kappa.unsqueeze(-1)*(torch.abs(X @ mu_unit.mH)**2).T
        return logpdf 

class ACG(PCMMtorchBaseModel):
    """ ACG distribution on the (complex) projective hyperplane.
    The ACG distribution is normally parameterized by a covariance matrix. Here, we have constructed 
    a low-rank-plus-diagonal approximation of the covariance matrix, which is parameterized by a rank r matrix M.
    The ACG distribution will fail if the input data vectors are not normalized to unit length.
    Args:
        p (int): dimensionality of the data vectors
        rank (int): rank of the low-rank approximation of the covariance matrix
        K (int): number of clusters
        HMM (bool): whether to use the HMM variant of the model (default: False)
        complex (bool): whether to use the complex ACG distribution (default: False)
        samples_per_sequence (int): number of samples per sequence to be used by HMM (default: 0, meaning one long sequence)
        params (dict): dictionary containing the parameters of the model, if available (default: None)
    """
    def __init__(self, p:int, rank:int, K:int=1, HMM:bool=False, complex:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = p
        self.r = rank
        self.K = K
        self.HMM = HMM
        if samples_per_sequence is None:
            samples_per_sequence = 0
        self.samples_per_sequence = torch.as_tensor(samples_per_sequence)
        self.distribution = 'ACG_lowrank'
        
        self.complex = complex
        if complex:
            self.a = torch.as_tensor(1)
            self.c = torch.as_tensor(self.p)
            self.distribution = 'Complex_'+self.distribution
        else:
            self.a = torch.as_tensor(0.5)
            self.c = torch.tensor(self.p/2)
        
        # precompute log-surface area of the unit hypersphere
        self.logSA_sphere = torch.lgamma(self.c) - torch.log(torch.as_tensor(2)) -self.c* torch.log(torch.as_tensor(math.pi))

        self.flag_normalized_input_data = False

        # initialize parameters
        if params is not None:
            self.unpack_params(params)
            
    def log_pdf(self,X, recompute_statics=False):
        # check for normalized input data
        if not self.flag_normalized_input_data:
            if not torch.allclose(torch.linalg.norm(X,axis=1),torch.as_tensor(1.,dtype=torch.double)):
                raise ValueError("For the Watson distribution, the input data vectors should be normalized to unit length.")
            else:
                self.flag_normalized_input_data = True

        # see supplementary material for the derivation of the logpdf for low-rank ACG
        D = torch.eye(self.r) + self.M.mH @ self.M
        v = torch.zeros(self.K, X.shape[0], dtype=X.dtype)

        # loop over components (is faster than batch matrix multiplication)
        for k in range(self.K):
            XM = torch.conj(X) @ self.M[k]
            v[k] = 1 - torch.sum(XM @ torch.linalg.inv(D[k]) * torch.conj(XM), dim=-1)

        log_pdf = self.logSA_sphere - self.a * torch.logdet(D).real.unsqueeze(-1) - self.c * torch.log(v.real)
        return log_pdf

class MACG(PCMMtorchBaseModel):
    def __init__(self, p:int, q:int, rank:int, K:int=1, HMM:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = p
        self.q = q
        self.r = rank
        self.K = K
        self.HMM = HMM
        if samples_per_sequence is None:
            samples_per_sequence = 0
        self.samples_per_sequence = torch.as_tensor(samples_per_sequence)
        self.distribution = 'MACG_lowrank'

        self.flag_normalized_input_data = False

        # initialize parameters
        if params is not None:
            self.unpack_params(params)

    def log_pdf(self,X, recompute_statics=False):
        if not self.flag_normalized_input_data:
            if torch.allclose(torch.linalg.norm(X[:,:,0],axis=1),torch.as_tensor(1.,dtype=torch.double))!=1:
                raise ValueError("For the MACG distribution, the input data vectors should be normalized to unit length (and orthonormal, but this is not checked).")
            else:
                self.flag_normalized_input_data = True        
        D = torch.swapaxes(self.M,-2,-1)@self.M+torch.eye(self.r)
        log_det_D = torch.logdet(D)
        
        v = torch.zeros(self.K, X.shape[0])
        for k in range(self.K):
            L, Q = torch.linalg.eigh(torch.linalg.inv(D[k]))
            D_sqrtinv = (Q * L.sqrt().unsqueeze(-2)) @ Q.mH
            XtM = X.mH@self.M[k].unsqueeze(0)
            S2 = torch.linalg.svdvals(XtM@D_sqrtinv.unsqueeze(0))
            v[k] = torch.sum(torch.log(1/(S2**2)-1),dim=-1)+2*torch.sum(torch.log(S2),dim=-1)

        log_pdf = - (self.q/2)*log_det_D.unsqueeze(-1) - self.p/2*v
        return log_pdf
    
class SingularWishart(PCMMtorchBaseModel):
    def __init__(self, p:int, q:int, rank:int, K:int=1, HMM:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = p
        self.q = q
        self.r = rank
        self.K = K
        self.HMM = HMM
        if samples_per_sequence is None:
            samples_per_sequence = 0
        self.samples_per_sequence = torch.as_tensor(samples_per_sequence)
        self.distribution = 'SingularWishart_lowrank'
        self.log_det_S11 = None
        
        loggamma_q = (self.q*(self.q-1)/4)*torch.log(torch.as_tensor(math.pi))+torch.sum(torch.lgamma(torch.as_tensor(self.q)-torch.arange(self.q)/2))
        self.log_norm_constant = self.q*(self.q-self.p)/2*torch.log(torch.as_tensor(math.pi))-self.p*self.q/2*torch.log(torch.as_tensor(2))-loggamma_q

        self.flag_normalized_input_data = False
        
        # initialize parameters
        if params is not None:
            self.unpack_params(params)

    def log_pdf(self,X, recompute_statics=False):
        if not self.flag_normalized_input_data:
            X_weights = torch.linalg.norm(X,axis=1)**2
            if not torch.allclose(torch.sum(X_weights,axis=1),torch.as_tensor(self.p,dtype=torch.double)):
                raise ValueError("In weighted grassmann clustering, the scale of the input data vectors should be equal to the square root of the eigenvalues. If the scale does not sum to the dimensionality, this error is thrown")
            else:
                self.flag_normalized_input_data = True

        # while Q_q^T Q_q != U_q^T L U_q, their determinants are the same
        if self.log_det_S11 is None:
            self.log_det_S11 = torch.logdet(torch.swapaxes(X[:,:self.q,:],-2,-1)@X[:,:self.q,:]).unsqueeze(0)
        if recompute_statics:
            log_det_S11 = torch.logdet(torch.swapaxes(X[:,:self.q,:],-2,-1)@X[:,:self.q,:]).unsqueeze(0)
        else:
            log_det_S11 = self.log_det_S11

        gamma = torch.nn.functional.softplus(self.gamma)

        M_tilde = self.M*torch.sqrt(1/gamma.unsqueeze(-1).unsqueeze(-1))

        D = torch.swapaxes(M_tilde,-2,-1)@M_tilde+torch.eye(self.r)
        log_det_D = self.p*torch.log(gamma)+torch.logdet(D)
        
        v = torch.zeros(self.K, X.shape[0])
        for k in range(self.K):
            L, Q = torch.linalg.eigh(torch.linalg.inv(D[k]))
            D_sqrtinv = (Q * L.sqrt().unsqueeze(-2)) @ Q.mH
            QtM_tilde = X.mH@M_tilde[k].unsqueeze(0)
            v[k] = 1/gamma[k]*(self.p - torch.linalg.norm(QtM_tilde@D_sqrtinv.unsqueeze(0),dim=(-2,-1))**2)
        
        log_pdf = self.log_norm_constant - (self.q/2)*log_det_D.unsqueeze(-1) + (self.q-self.p-1)/2*log_det_S11 - 1/2*v
        return log_pdf

class Normal(PCMMtorchBaseModel):
    def __init__(self, p:int, rank:int, K:int=1, complex:bool=False, HMM:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = p
        self.r = rank
        self.K = K
        self.HMM = HMM
        if samples_per_sequence is None:
            samples_per_sequence = 0
        self.samples_per_sequence = torch.as_tensor(samples_per_sequence)
        self.distribution = 'Normal_lowrank'
        self.complex = complex

        if complex:
            self.a = torch.as_tensor(1)
            self.c = torch.as_tensor(self.p)
            self.distribution = 'Complex_'+self.distribution
        else:
            self.a = torch.as_tensor(0.5)
            self.c = torch.as_tensor(self.p/2)
        
        self.log_norm_constant = -self.c*torch.log(1/self.a*torch.as_tensor(math.pi))
        self.norm_x = None

        # initialize parameters
        if params is not None:
            self.unpack_params(params)

    def log_pdf(self,X, recompute_statics=False):
        if self.norm_x is None:
            self.norm_x = (torch.linalg.norm(X,dim=1)**2).unsqueeze(0)
        if recompute_statics:
            norm_x = (torch.linalg.norm(X,dim=1)**2).unsqueeze(0)
        else:
            norm_x = self.norm_x


        gamma = torch.nn.functional.softplus(self.gamma)

        M_tilde = self.M*torch.sqrt(1/gamma.unsqueeze(-1).unsqueeze(-1))

        D = M_tilde.mH @ M_tilde+torch.eye(self.r)
        log_det_D = self.p*torch.log(gamma)+torch.logdet(D)
        
        v = torch.zeros(self.K, X.shape[0])
        for k in range(self.K):
            L, Q = torch.linalg.eigh(torch.linalg.inv(D[k]))
            D_sqrtinv = (Q * L.sqrt().unsqueeze(-2)) @ Q.mH
            XtM_tilde = torch.conj(X).unsqueeze(-2)@M_tilde[k].unsqueeze(0)
            v[k] = 1/gamma[k]*(norm_x - torch.linalg.norm(XtM_tilde@D_sqrtinv.unsqueeze(0),dim=(-2,-1))**2)
        
        log_pdf = self.log_norm_constant - self.a*log_det_D.real.unsqueeze(-1) - self.a*v
        return log_pdf