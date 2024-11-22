import numpy as np
from PCMM.PCMM_EM.PCMMEMBaseModel import PCMMEMBaseModel

class Normal(PCMMEMBaseModel):
    def __init__(self, p:int, rank:int=None, K:int=1, complex:bool=False, params:dict=None):
        super().__init__()

        self.K = K
        self.p = p
        if rank is None or rank==0: #rank zero means the fullrank version
            self.r = p
            self.distribution = 'Normal_fullrank'
        else: #the lowrank version can also be full rank
            self.r = rank
            self.distribution = 'Normal_lowrank'

        if complex:
            self.a = 1
            self.c = self.p
            self.distribution = 'Complex_'+self.distribution
        else:
            self.a = 0.5
            self.c = self.p/2
        
        self.log_norm_constant = -self.c*np.log(1/self.a*np.pi)
        self.norm_x = None

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)
    
    def log_pdf_lowrank(self, X):

        M_tilde = self.M*np.sqrt(1/np.atleast_1d(self.gamma)[:,None,None])
        
        _,S1,V1t = np.linalg.svd(M_tilde)
        log_det_D = self.p*np.log(self.gamma)+np.sum(np.log(1/S1**2+1),axis=-1)+2*np.sum(np.log(S1),axis=-1)
        D_sqrtinv = (np.swapaxes(V1t.conj(),-2,-1)*np.sqrt(1/(1+S1**2))[:,None])@V1t

        if self.norm_x is None:
            self.norm_x = np.linalg.norm(X,axis=1)**2
        
        v = 1/np.atleast_1d(self.gamma)[:,None]*(self.norm_x[None,:] - np.linalg.norm(X.conj()[None,:,None,:]@M_tilde[:,None,:,:]@D_sqrtinv[:,None,:,:],axis=(-2,-1))**2)
        log_pdf = self.log_norm_constant - self.a*np.atleast_1d(log_det_D)[:,None] - self.a*np.real(v)

        return log_pdf

    def log_pdf_fullrank(self, X):
        XtLX = np.real(np.sum(X.conj()@np.linalg.inv(self.Lambda)*X,axis=-1)) #should be real
        log_det_L = np.atleast_1d(np.real(self.logdet(self.Lambda)))[:,None] #should be real
        log_pdf = self.log_norm_constant - self.a * log_det_L -self.a*XtLX
        return log_pdf

    def log_pdf(self, X):
        if 'lowrank' in self.distribution:
            return self.log_pdf_lowrank(X)
        elif 'fullrank' in self.distribution:
            return self.log_pdf_fullrank(X)
        
    def M_step_single_component(self,X,beta,M=None,gamma=None,max_iter=int(1e8),tol=1e-10):
            
        if 'lowrank' in self.distribution:

            loss = []
            
            if gamma is None:
                raise ValueError("gamma is not provided")

            if self.norm_x is None:
                norm_x = np.linalg.norm(X,axis=1)**2
            else:
                norm_x = self.norm_x
            
            M_tilde = M * np.sqrt(1/gamma)
            # _,S1,V1t = np.linalg.svd(M_tilde)
            S1 = np.linalg.svd(M_tilde,full_matrices=False,compute_uv=False)
            # D_inv = V1t.T.conj()@np.diag(1/(1+S1**2))@V1t
            D_inv = np.linalg.inv(M_tilde.conj().T@M_tilde+np.eye(self.r))

            trZt_oldZ_old = gamma**2*(np.sum(S1**4)+2*np.sum(S1**2)+self.p)
            gamma_old = gamma
            S1_old = S1
            M_tilde_old = M_tilde

            #precompute
            beta_sum = np.sum(beta)
            beta_S = (beta[:,None]*X).T@X.conj()
            beta_x_norm_sum = np.sum(beta*norm_x)

            for j in range(max_iter):

                # M_tilde update
                M_tilde = 1/(gamma*beta_sum)*beta_S@M_tilde@D_inv
                # _,S1,V1t = np.linalg.svd(M_tilde)
                S1 = np.linalg.svd(M_tilde,full_matrices=False,compute_uv=False)
                # D_inv = V1t.T.conj()@np.diag(1/(1+S1**2))@V1t
                D_inv = np.linalg.inv(M_tilde.conj().T@M_tilde+np.eye(self.r))
                #gamma update
                gamma = 1/(self.p*beta_sum)*(beta_x_norm_sum-np.trace(M_tilde@D_inv@M_tilde.T.conj()@beta_S).real)

                # convergence criterion
                trZtZ = gamma**2*(np.sum(S1**4)+2*np.sum(S1**2)+self.p)
                trZtZt_old = gamma*gamma_old*(np.linalg.norm(M_tilde.T.conj()@M_tilde_old)**2 + np.sum(S1**2) + np.sum(S1_old**2)+self.p)
                loss.append(trZtZ+trZt_oldZ_old-2*trZtZt_old)
                
                if j>0:
                    if loss[-1]<tol:
                        # if np.any(np.array(loss)<tol):
                            # if loss[-1]<-1e-5:
                            #     raise Warning("Loss is negative. Check M_step_single_component")
                            # raise Warning("Loss is negative. Check M_step_single_component")
                        break
                
                # To measure convergence
                trZt_oldZ_old = trZtZ
                gamma_old = gamma
                M_tilde_old = M_tilde
                S1_old = S1

            if j==max_iter-1:
                raise Warning("M-step did not converge")

            # output M, not M_tilde
            M = M_tilde*np.sqrt(gamma)
            # print(j,np.linalg.norm(M),gamma,np.sum(beta))
            return M, gamma
        elif 'fullrank' in self.distribution:
            Psi = 1/(np.sum(beta))*(beta[:,None]*X).T@X
            return Psi

    def M_step(self,X):
        if self.K!=1:
            beta = np.exp(self.log_density-self.logsum_density)
            self.update_pi(beta)
        else:
            beta = np.ones((1,X.shape[0]))
        for k in range(self.K):
            if 'lowrank' in self.distribution:
                self.M[k],self.gamma[k] = self.M_step_single_component(X,beta[k],self.M[k],gamma=self.gamma[k])
            elif 'fullrank' in self.distribution:
                self.Lambda[k] = self.M_step_single_component(X,beta[k])