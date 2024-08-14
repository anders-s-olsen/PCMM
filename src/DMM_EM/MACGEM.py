import numpy as np
from src.DMM_EM.DMMEMBaseModel import DMMEMBaseModel

class MACG(DMMEMBaseModel):
    def __init__(self, p:int,q:int, K:int=1, rank=None, params:dict=None):
        super().__init__()

        self.K = K
        self.p = p
        self.q = q
        if rank is None or rank==0: #rank zero means the fullrank version
            self.r = p
            self.distribution = 'MACG_fullrank'
        else: #the lowrank version can also be full rank
            self.r = rank
            self.distribution = 'MACG_lowrank'

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)
    
    def log_pdf_lowrank(self, X):

        _,S1,V1t = np.linalg.svd(self.M,full_matrices=False)
        log_det_D = np.sum(np.log(1/S1**2+1),axis=-1)+2*np.sum(np.log(S1),axis=-1)

        D_sqrtinv = (np.swapaxes(V1t,-2,-1)*np.sqrt(1/(1+S1**2))[:,None])@V1t
        XtM = np.swapaxes(X,-2,-1)[None,:,:,:]@self.M[:,None,:,:]
        S2 = np.linalg.svd(XtM@(D_sqrtinv[:,None]),compute_uv=False)
        v = np.sum(np.log(1/(S2**2)-1),axis=-1)+2*np.sum(np.log(S2),axis=-1)
        
        log_pdf = - (self.q/2)*np.atleast_1d(log_det_D)[:,None] - self.p/2*v
        return log_pdf

    def log_pdf_fullrank(self, X):
        logdetXtLX = self.logdet(np.swapaxes(X,-2,-1)[None,:,:,:]@np.linalg.inv(self.Lambda)[:,None,:,:]@X)
        log_pdf = - (self.q/2)*np.atleast_1d(self.logdet(self.Lambda))[:,None] - self.p/2*logdetXtLX
        return log_pdf

    def log_pdf(self, X):
        if self.distribution == 'MACG_lowrank':
            return self.log_pdf_lowrank(X)
        elif self.distribution == 'MACG_fullrank':
            return self.log_pdf_fullrank(X)
        
    def M_step_single_component(self,X,beta,M=None,Lambda=None,max_iter=int(1e5),tol=1e-8):
            
        if self.distribution == 'MACG_lowrank':
            Q = beta[:,None,None]*X

            loss = []
            o = np.linalg.norm(M,'fro')**2
            gamma = 1/(1+o/self.p)
            M = np.sqrt(gamma)*M
            _,S1,V1t = np.linalg.svd(M,full_matrices=False)
            
            trZt_oldZ_old = np.sum(S1**4)+2*gamma*np.sum(S1**2)+gamma**2*self.p
            gamma_old = gamma
            S1_old = S1
            M_old = M.copy()

            for j in range(max_iter):

                XtM = np.swapaxes(X,-2,-1)@M
                D_sqrtinv = V1t.T@np.diag(np.sqrt(1/(1+S1**2)))@V1t
                U2,S2,V2t = np.linalg.svd(XtM@D_sqrtinv,full_matrices=False)
                M = self.p/(self.q*np.sum(beta))*np.sum(Q@(U2*(S2/(1-S2**2))[:,None,:])@V2t,axis=0)@D_sqrtinv

                # Then we trace-normalize M
                o = np.linalg.norm(M,'fro')**2
                gamma = 1/(1+o/self.p)
                M = np.sqrt(gamma)*M

                # To measure convergence, we compute norm(Z-Z_old)**2
                _,S1,V1t = np.linalg.svd(M,full_matrices=False)
                trZtZ = np.sum(S1**4)+2*gamma*np.sum(S1**2)+gamma**2*self.p
                trZtZt_old = np.linalg.norm(M.T@M_old)**2 + gamma_old*gamma*self.p + gamma_old*np.sum(S1**2) + gamma*np.sum(S1_old**2)
                loss.append(trZtZ+trZt_oldZ_old-2*trZtZt_old)
                
                if j>0:
                    if loss[-1]<tol:
                        if np.any(np.array(loss)<-tol):
                            raise Warning("Loss is negative. Check M_step_single_component")
                        break
                
                # To measure convergence
                trZt_oldZ_old = trZtZ
                gamma_old = gamma
                M_old = M
                S1_old = S1

            if j==max_iter-1:
                raise Warning("M-step did not converge")
            
            # output the unnormalized M such that the log_pdf is computed without having to include the normalization factor (pdf is scale invariant)
            M = M/np.sqrt(gamma)
            # print(j,np.linalg.norm(M),gamma,np.sum(beta))
            return M
        elif self.distribution == 'MACG_fullrank':
            Q = beta[:,None,None]*X

            for j in range(max_iter):

                # this has been tested in the "Naive" version below
                XtLX = np.swapaxes(X,-2,-1)@np.linalg.inv(Lambda)@X
                L = np.linalg.eigvalsh(XtLX)
                XtLX_trace = np.sum(1/L,axis=1) #trace of inverse is sum of inverse eigenvalues
                
                Lambda = self.p*np.sum(Q@np.linalg.inv(XtLX)@np.swapaxes(X,-2,-1),axis=0) \
                    /(np.sum(beta*XtLX_trace))
                
                if j>0:
                    if np.linalg.norm(Lambda_old-Lambda)<tol:
                        break
                Lambda_old = Lambda
            return Lambda

    def M_step(self,X):
        beta = np.exp(self.log_density-self.logsum_density)
        self.update_pi(beta)
        for k in range(self.K):
            if self.distribution == 'MACG_lowrank':
                self.M[k] = self.M_step_single_component(X,beta[k],self.M[k],None)
            elif self.distribution == 'MACG_fullrank':
                self.Lambda[k] = self.M_step_single_component(X,beta[k],None,self.Lambda[k])
