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
        # from time import time
        # t0 = time()
        # D = np.eye(self.r) + np.swapaxes(self.M,-2,-1)@self.M 
        # XtM = np.swapaxes(X,-2,-1)[None,:,:,:]@self.M[:,None,:,:]
        # log_det_D = self.logdet(D)
        # v = self.logdet(D[:,None]-np.swapaxes(XtM,-2,-1)@XtM)-log_det_D[:,None]
        # t1 = time()

        # t2 = time()
        XtM = np.swapaxes(X,-2,-1)[None,:,:,:]@self.M[:,None,:,:]
        _,S1,V1t = np.linalg.svd(self.M,full_matrices=False)
        D_sqrtinv = (np.swapaxes(V1t,-2,-1)*np.sqrt(1/(1+S1**2))[:,None])@V1t
        S2 = np.linalg.svdvals(XtM@(D_sqrtinv[:,None]))
        v = np.sum(np.log(1/(S2**2)-1),axis=-1)+2*np.sum(np.log(S2),axis=-1)
        log_det_D = np.sum(np.log(1/S1**2+1),axis=-1)+2*np.sum(np.log(S1),axis=-1)
        # t3 = time()
        # print(t1-t0,t3-t2)
        
        log_pdf = self.logSA_stiefel - (self.q/2)*log_det_D[:,None] - self.half_p*v
        return log_pdf

    def log_pdf_fullrank(self, X):
        logdetXtLX = self.logdet(np.swapaxes(X,-2,-1)[None,:,:,:]@np.linalg.inv(self.Lambda)[:,None,:,:]@X)
        log_pdf = self.logSA_stiefel - (self.q/2)*self.logdet(self.Lambda)[:,None] - self.half_p*logdetXtLX
        return log_pdf

    def log_pdf(self, X,L=None):
        if self.distribution == 'MACG_lowrank':
            return self.log_pdf_lowrank(X)
        elif self.distribution == 'MACG_fullrank':
            return self.log_pdf_fullrank(X)
        
    def M_step_single_component(self,X,Beta,M=None,Lambda=None,max_iter=int(1e5),tol=1e-6):
        # if n<(p*(p-1)*q):
        #     Warning("Too high dimensionality compared to number of observations. Sigma cannot be estimated")
            
        if self.distribution == 'MACG_lowrank':
            Q = Beta[:,None,None]*X

            loss = []
            o = np.linalg.norm(M,'fro')**2
            b = 1/(1+o/self.p)
            M = np.sqrt(b)*M
            _,S1,V1t = np.linalg.svd(M)
            
            trZt_oldZ_old = np.sum(S1**4)+2*b*np.sum(S1**2)+b**2*self.p
            b_old = b
            S1_old = S1
            M_old = M.copy()

            for j in range(max_iter):
                # from time import time
                # t0 = time()

                # # Woodbury scaled. First we update M
                # D_inv = np.linalg.inv(np.eye(self.r)+MtM)
                # XtM = np.swapaxes(X,-2,-1)@M
                
                # # This is okay, because the matrices to be inverted are only 2x2
                # V_inv = np.linalg.inv(np.eye(self.q)-XtM@D_inv@np.swapaxes(XtM,-2,-1))
                
                # # Works, think it's okay in terms of speed bco precomputation of XtM
                # M1 = self.p/(self.q*np.sum(Beta))*np.sum(Q@V_inv@XtM,axis=0)@(np.eye(self.r)-D_inv@M.T@M)
                # t1 = time()

                XtM = np.swapaxes(X,-2,-1)@M
                D_sqrtinv = V1t.T@np.diag(np.sqrt(1/(1+S1**2)))@V1t
                U2,S2,V2t = np.linalg.svd(XtM@D_sqrtinv,full_matrices=False)
                M = self.p/(self.q*np.sum(Beta))*np.sum(Q@(U2*(S2/(1-S2**2))[:,None,:])@V2t,axis=0)@D_sqrtinv

                # Then we trace-normalize M
                o = np.linalg.norm(M,'fro')**2
                b = 1/(1+o/self.p)
                # c = b*self.p/self.r
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
        elif self.distribution == 'MACG_fullrank':
            Q = Beta[:,None,None]*X

            for j in range(max_iter):

                # this has been tested in the "Naive" version below
                XtLX = np.swapaxes(X,-2,-1)@np.linalg.inv(Lambda)@X
                L,_ = np.linalg.eigh(XtLX)
                XtLX_trace = np.sum(1/L,axis=1) #trace of inverse is sum of inverse eigenvalues
                
                Lambda = self.p*np.sum(Q@np.linalg.inv(XtLX)@np.swapaxes(X,-2,-1),axis=0) \
                    /(np.sum(Beta*XtLX_trace))
                
                if j>0:
                    if np.linalg.norm(Lambda_old-Lambda)<tol:
                        break
                Lambda_old = Lambda
            return Lambda

    def M_step(self,X,L=None):
        Beta = np.exp(self.log_density-self.logsum_density)
        self.update_pi(Beta)
        for k in range(self.K):
            if self.distribution == 'MACG_lowrank':
                self.M[k] = self.M_step_single_component(X,Beta[k],self.M[k],None)
            elif self.distribution == 'MACG_fullrank':
                self.Lambda[k] = self.M_step_single_component(X,Beta[k],None,self.Lambda[k])