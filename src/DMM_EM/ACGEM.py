import numpy as np
from src.DMM_EM.DMMEMBaseModel import DMMEMBaseModel
from scipy.special import loggamma
import time
# import multiprocessing as mp
# import concurrent.futures
# import numba as nb
# @nb.njit(nb.float64[:,:,:](nb.float64[:,::1],nb.float64[:,:,:],nb.float64[:,:]),cache=True)

class ACG(DMMEMBaseModel):
    def __init__(self, p:int, K:int=1, rank=None, params:dict=None):
        super().__init__()

        self.K = K
        self.p = p
        self.half_p = self.p/2
        if rank is None or rank==0: #rank zero means the fullrank version
            self.r = p
            self.distribution = 'ACG_fullrank'
        else: #the lowrank version can also be full rank
            self.r = rank
            self.distribution = 'ACG_lowrank'
        
        # precompute log-surface area of the unit hypersphere
        self.logSA_sphere = loggamma(self.half_p) - np.log(2) -self.half_p* np.log(np.pi)

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)
    
    def log_pdf_lowrank(self, X):
        _,S1,V1t = np.linalg.svd(self.M,full_matrices=False)
        D_inv = (np.swapaxes(V1t,-2,-1)*(1/(1+S1**2))[:,None])@V1t
        # D = np.eye(self.r) + np.swapaxes(self.M,-2,-1)@self.M 
        log_det_D = np.sum(np.log(1/S1**2+1),axis=-1)+2*np.sum(np.log(S1),axis=-1)
        XM = X[None,:,:]@self.M
        # v = 1-np.sum(XM@np.linalg.inv(D)*XM,axis=-1) 
        v = 1-np.sum(XM@D_inv*XM,axis=-1) 
        # log_pdf = self.logSA_sphere - 0.5 * self.logdet(D)[:,None] - self.half_p * np.log(v)
        log_pdf = self.logSA_sphere - 0.5 * log_det_D[:,None] - self.half_p * np.log(v)
        return log_pdf
    
    def log_pdf_fullrank(self, X):
        XtLX = np.sum(X@np.linalg.inv(self.Lambda)*X,axis=2)
        log_pdf = self.logSA_sphere - 0.5 * self.logdet(self.Lambda)[:,None] -self.half_p*np.log(XtLX)
        return log_pdf

    def log_pdf(self, X,L=None):
        if self.distribution == 'ACG_lowrank':
            return self.log_pdf_lowrank(X)
        elif self.distribution == 'ACG_fullrank':
            return self.log_pdf_fullrank(X)

    def M_step_single_component(self,X,Beta,M=None,Lambda=None,max_iter=int(1e5),tol=1e-6):
        n,p = X.shape
        if n<p*(p-1):
            Warning("Too high dimensionality compared to number of observations. Lambda cannot be calculated")
        if self.distribution == 'ACG_lowrank':
            Q = (Beta[:,None]*X).T

            loss = []
            o_all = []
            o = np.linalg.norm(M,'fro')**2
            o_all.append(o)
            b = 1/(1+o/p)
            M = np.sqrt(b)*M #we control the scale of M to avoid numerical issues
            _,S1,V1t = np.linalg.svd(M,full_matrices=False)
            
            trZt_oldZ_old = np.sum(S1**4)+2*b*np.sum(S1**2)+b**2*self.p
            b_old = b
            S1_old = S1
            M_old = M.copy()

            for j in range(max_iter):

                # First we update M

                D_inv = V1t.T@np.diag(1/(1+S1**2))@V1t
                XM = X@M
                XMD_inv = XM@D_inv
                v = 1-np.sum(XMD_inv*XM,axis=1) #denominator
                M = p/np.sum(Beta)*Q@(XMD_inv/v[:,None])

                # Then we trace-normalize M
                o = np.linalg.norm(M,'fro')**2
                o_all.append(o)
                b = 1/(1+o/p)
                M = np.sqrt(b)*M

                # To measure convergence, we compute norm(Z-Z_old)**2
                _,S1,V1t = np.linalg.svd(M,full_matrices=False)
                trZtZ = np.sum(S1**4)+2*b*np.sum(S1**2)+b**2*self.p
                trZtZt_old = np.linalg.norm(M.T@M_old)**2 + b_old*b*self.p + b_old*np.sum(S1**2) + b*np.sum(S1_old**2)
                loss.append(trZtZ+trZt_oldZ_old-2*trZtZt_old)
                
                if j>0:
                    if loss[-1]<tol:
                        if np.any(np.array(loss)<0):
                            raise Warning("Loss is negative. Check M_step_single_component")
                        break
                
                # To measure convergence
                trZt_oldZ_old = trZtZ
                b_old = b
                M_old = M
                S1_old = S1
            return M
        elif self.distribution == 'ACG_fullrank':
            Lambda_old = np.eye(self.p)
            Q = Beta[:,None]*X
            for j in range(max_iter):
                XtLX = np.sum(X@np.linalg.inv(Lambda)*X,axis=1)
                Lambda = p/np.sum(Beta/XtLX)*(Q.T/XtLX)@X
                if j>0:
                    if np.linalg.norm(Lambda_old-Lambda)<tol:
                        break
                Lambda_old = Lambda
            return Lambda

    def M_step(self,X,L=None):
        Beta = np.exp(self.log_density-self.logsum_density)
        self.update_pi(Beta)
        for k in range(self.K):
            if self.distribution == 'ACG_lowrank':
                self.M[k] = self.M_step_single_component(X,Beta[k],self.M[k],None)
            elif self.distribution == 'ACG_fullrank':
                self.Lambda[k] = self.M_step_single_component(X,Beta[k],None,self.Lambda[k])
