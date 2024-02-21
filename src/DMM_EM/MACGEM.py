import numpy as np
from src.DMM_EM.DMMEMBaseModel import DMMEMBaseModel
from scipy.special import loggamma

class MACG(DMMEMBaseModel):
    def __init__(self, p:int,q:int, K:int=1, rank=None, params:dict=None):
        super().__init__()

        self.K = K
        self.p = p
        self.q = q
        self.half_p = self.p/2
        if rank is None or rank==0: #rank zero means the fullrank version
            self.r = p
            self.distribution = 'MACG_fullrank'
        else: #the lowrank version can also be full rank
            self.r = rank
            self.distribution = 'MACG_lowrank'
        
        # precompute log-surface area of the Stiefel manifold
        loggamma_k = (self.q*(self.q-1)/4)*np.log(np.pi)+np.sum(loggamma(self.half_p-np.arange(self.q)/2))
        self.logSA_stiefel = loggamma_k-self.q*np.log(2)-self.q*self.half_p*np.log(np.pi)

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)
    
    def log_pdf_lowrank(self, X):
        D = np.eye(self.r) + np.swapaxes(self.M,-2,-1)@self.M 
        log_det_D = self.logdet(D)
        XtM = np.swapaxes(X,-2,-1)[None,:,:,:]@self.M[:,None,:,:]
        v = self.logdet(D[:,None]-np.swapaxes(XtM,-2,-1)@XtM)-log_det_D[:,None]
        
        log_pdf = self.logSA_stiefel - (self.q/2)*self.logdet(D)[:,None] - self.half_p*v
        return log_pdf

    def log_pdf_fullrank(self, X):
        logdetXtLX = self.logdet(np.swapaxes(X,-2,-1)[None,:,:,:]@np.linalg.inv(self.Lambda)[:,None,:,:]@X)
        log_pdf = self.logSA_stiefel - (self.q/2)*self.logdet(self.Lambda)[:,None] - self.half_p*logdetXtLX
        return log_pdf

    def log_pdf(self, X):
        if self.distribution == 'MACG_lowrank':
            return self.log_pdf_lowrank(X)
        elif self.distribution == 'MACG_fullrank':
            return self.log_pdf_fullrank(X)
        
    def M_step_lowrank(self,X,max_iter=int(1e5),tol=1e-6):
        n,p,q = X.shape
        if n<(p*(p-1)*q):
            Warning("Too high dimensionality compared to number of observations. Sigma cannot be estimated")
        
        Beta = np.exp(self.log_density-self.logsum_density)
        self.update_pi(Beta)
        for k in range(self.K):
            M = self.M[k]
            Q = Beta[k,:,None,None]*X

            loss = []
            o = np.linalg.norm(M,'fro')**2
            b = 1/(1+o/p)
            M = np.sqrt(b)*M
            
            trMMtMMt_old = np.trace(M.T@M@M.T@M)
            M_old = M

            for j in range(max_iter):

                # Woodbury scaled. First we update M
                MtM = M.T@M
                D_inv = np.linalg.inv(np.eye(self.r)+MtM)
                XtM = np.swapaxes(X,-2,-1)@M
                
                # This is okay, because the matrices to be inverted are only 2x2
                V_inv = np.linalg.inv(np.eye(q)-XtM@D_inv@np.swapaxes(XtM,-2,-1))
                
                # Works, think it's okay in terms of speed bco precomputation of XtM
                M = p/(q*np.sum(Beta[k]))*np.sum(Q@V_inv@XtM,axis=0)@(np.eye(self.r)-D_inv@MtM)

                # Then we trace-normalize M
                o = np.linalg.norm(M,'fro')**2
                b = 1/(1+o/p)
                M = np.sqrt(b)*M

                # To measure convergence, we compute norm(Z-Z_old)**2
                trMMtMMt = np.trace(M.T@M@M.T@M)
                loss.append(trMMtMMt+trMMtMMt_old-2*np.trace(M@M.T@M_old@M_old.T))
                
                if j>0:
                    if loss[-1]<tol:
                        break
                
                # To measure convergence
                trMMtMMt_old = trMMtMMt
                M_old = M
            self.M[k] = M
        
    def M_step_fullrank(self,X,max_iter=int(1e5),tol=1e-6):
        n,p,q = X.shape
        if n<(p*(p-1)*q):
            Warning("Too high dimensionality compared to number of observations. Sigma cannot be estimated")
        
        Beta = np.exp(self.log_density-self.logsum_density)
        self.update_pi(Beta)
        for k in range(self.K):
            Lambda = self.Lambda[k]
            Q = Beta[k,:,None,None]*X

            for j in range(max_iter):

                # this has been tested in the "Naive" version below
                XtLX = np.swapaxes(X,-2,-1)@np.linalg.inv(Lambda)@X
                L,V = np.linalg.eigh(XtLX)
                XtLX_trace = np.sum(1/L,axis=1) #trace of inverse is sum of inverse eigenvalues
                
                Lambda = p*np.sum(Q@np.linalg.inv(XtLX)@np.swapaxes(X,-2,-1),axis=0) \
                    /(np.sum(Beta[k]*XtLX_trace))
                
                if j>0:
                    if np.linalg.norm(Lambda_old-Lambda)<tol:
                        break
                Lambda_old = Lambda
            self.Lambda[k] = Lambda

    def M_step(self,X):
        if self.distribution == 'MACG_lowrank':
            self.M_step_lowrank(X)
        elif self.distribution == 'MACG_fullrank':
            self.M_step_fullrank(X)