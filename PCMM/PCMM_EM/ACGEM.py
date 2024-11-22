import numpy as np
from PCMM.PCMM_EM.PCMMEMBaseModel import PCMMEMBaseModel
from scipy.special import loggamma

class ACG(PCMMEMBaseModel):
    def __init__(self, p:int, rank:int=None, K:int=1, complex:bool=False, params:dict=None):
        super().__init__()

        self.K = K
        self.p = p
        if rank is None or rank==0: #rank zero means the fullrank version
            self.r = p
            self.distribution = 'ACG_fullrank'
        else: #the lowrank version can also be full rank
            self.r = rank
            self.distribution = 'ACG_lowrank'

        if complex:
            self.a = 1
            self.c = self.p
            self.distribution = 'Complex_'+self.distribution
        else:
            self.a = 0.5
            self.c = self.p/2
        
        # precompute log-surface area of the unit hypersphere
        self.logSA_sphere = loggamma(self.c) - np.log(2) -self.c* np.log(np.pi)

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)
    
    def log_pdf_lowrank(self, X):
        _,S1,V1h = np.linalg.svd(self.M,full_matrices=False)
        log_det_D = np.sum(np.log(1/S1**2+1),axis=-1)+2*np.sum(np.log(S1),axis=-1)
        D_inv = (np.swapaxes(V1h.conj(),-2,-1)*(1/(1+S1**2))[:,None])@V1h

        XM = X[None,:,:].conj()@self.M #check the conj here and below
        v = 1-np.sum(XM@D_inv*XM.conj(),axis=-1) 

        log_pdf = self.logSA_sphere - self.a * np.atleast_1d(log_det_D)[:,None] - self.c * np.log(np.real(v))
        return log_pdf
    
    def log_pdf_fullrank(self, X):
        XtLX = np.real(np.sum(X.conj()@np.linalg.inv(self.Lambda)*X,axis=-1)) #should be real
        log_det_L = np.atleast_1d(np.real(self.logdet(self.Lambda)))[:,None] #should be real
        log_pdf = self.logSA_sphere - self.a * log_det_L -self.c*np.log(XtLX)
        return log_pdf

    def log_pdf(self, X):
        if 'lowrank' in self.distribution:
            return self.log_pdf_lowrank(X)
        elif 'fullrank' in self.distribution:
            return self.log_pdf_fullrank(X)

    def M_step_single_component(self,X,beta,M=None,Lambda=None,max_iter=int(1e5),tol=1e-10):
        n,p = X.shape
        if n<p*(p-1):
            Warning("Too high dimensionality compared to number of observations. Lambda cannot be calculated")
        if self.distribution in ['ACG_lowrank','Complex_ACG_lowrank']:
            Q = (beta[:,None]*X).T

            loss = []
            o = np.linalg.norm(M,'fro')**2
            gamma = 1/(1+o/self.p)
            # M_tilde = M*np.sqrt(gamma)
            M = np.sqrt(gamma)*M #we control the scale of M to avoid numerical issues
            M_tilde = M
            S1 = np.linalg.svd(M_tilde,full_matrices=False,compute_uv=False)
            
            trZt_oldZ_old = np.sum(S1**4)+2*gamma*np.sum(S1**2)+gamma**2*self.p
            gamma_old = gamma
            S1_old = S1
            # M_old = M.copy()
            M_tilde_old = M_tilde.copy()

            for j in range(max_iter):

                # First we update M
                D_inv = np.linalg.inv(M.conj().T@M+np.eye(self.r))
                # D_inv = V1h.conj().T@np.diag(1/(1+S1**2))@V1h
                XM = X.conj()@M 
                XMD_inv = XM@D_inv
                v = 1-np.sum(XMD_inv*XM.conj(),axis=1) #denominator
                M = self.p/np.sum(beta)*Q@(XMD_inv/v[:,None])

                # Then we trace-normalize M
                o = np.linalg.norm(M,'fro')**2
                gamma = 1/(1+o/self.p)

                # To measure convergence, we compute norm(Z-Z_old)**2
                # M_tilde = M*np.sqrt(gamma)
                M = np.sqrt(gamma)*M
                M_tilde = M
                S1 = np.linalg.svd(M_tilde,full_matrices=False,compute_uv=False)
                trZtZ = np.sum(S1**4)+2*gamma*np.sum(S1**2)+gamma**2*self.p
                trZtZt_old = np.linalg.norm(M_tilde.conj().T@M_tilde_old)**2 + gamma_old*gamma*self.p + gamma_old*np.sum(S1**2) + gamma*np.sum(S1_old**2)
                loss.append(trZtZ+trZt_oldZ_old-2*trZtZt_old)
                
                if j>0:
                    if loss[-1]<tol:
                        if np.any(np.array(loss)<-tol):
                            raise Warning("Loss is negative. Check M_step_single_component")
                        break
                
                # # # To measure convergence
                trZt_oldZ_old = trZtZ
                gamma_old = gamma
                # M_old = M
                S1_old = S1
                M_tilde_old = M_tilde

            if j==max_iter-1:
                raise Warning("M-step did not converge")

            # output the unnormalized M such that the log_pdf is computed without having to include the normalization factor (pdf is scale invariant)
            M = M/np.sqrt(gamma)
            # print(j,np.linalg.norm(M),gamma,np.sum(beta))
            return M
        elif self.distribution in ['ACG_fullrank','Complex_ACG_fullrank']:
            Lambda_old = Lambda.copy()
            Q = (beta[:,None]*X).T
            for j in range(max_iter):
                XtLX = np.sum(X.conj()@np.linalg.inv(Lambda)*X,axis=1)
                Lambda = self.p/np.sum(beta/XtLX)*(Q/XtLX)@X.conj()
                if j>0:
                    if np.linalg.norm(Lambda_old-Lambda)<tol:
                        break
                Lambda_old = Lambda
            return Lambda

    def M_step(self,X):
        if self.K!=1:
            beta = np.exp(self.log_density-self.logsum_density)
            self.update_pi(beta)
        else:
            beta = np.ones((1,X.shape[0]))
        for k in range(self.K):
            if self.distribution in ['ACG_lowrank','Complex_ACG_lowrank']:
                self.M[k] = self.M_step_single_component(X,beta[k],self.M[k],None)
            elif self.distribution in ['ACG_fullrank','Complex_ACG_fullrank']:
                self.Lambda[k] = self.M_step_single_component(X,beta[k],None,self.Lambda[k])