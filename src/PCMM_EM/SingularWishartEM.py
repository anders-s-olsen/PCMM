import numpy as np
from scipy.special import loggamma
from src.PCMM_EM.PCMMEMBaseModel import PCMMEMBaseModel

class SingularWishart(PCMMEMBaseModel):
    def __init__(self, p:int,q:int, K:int=1, rank=None, params:dict=None):
        super().__init__()

        self.K = K
        self.p = p
        self.q = q
        self.half_p = self.p/2
        if rank is None or rank==0: #rank zero means the fullrank version
            self.r = p
            self.distribution = 'SingularWishart_fullrank'
        else: #the lowrank version can also be full rank
            self.r = rank
            self.distribution = 'SingularWishart_lowrank'

        loggamma_k = (self.q*(self.q-1)/4)*np.log(np.pi)+np.sum(loggamma(self.q-np.arange(self.q)/2))
        self.log_norm_constant = self.q*(self.q-self.p)/2*np.log(np.pi)-self.p*self.q/2*np.log(2)-loggamma_k

        self.log_det_S11 = None

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)
    
    def log_pdf_lowrank(self, Q):

        M_tilde = self.M*np.sqrt(1/np.atleast_1d(self.gamma)[:,None,None])
        
        _,S1,V1t = np.linalg.svd(M_tilde)
        log_det_D = self.p*np.log(self.gamma)+np.sum(np.log(1/S1**2+1),axis=-1)+2*np.sum(np.log(S1),axis=-1)
        D_sqrtinv = (np.swapaxes(V1t,-2,-1)*np.sqrt(1/(1+S1**2))[:,None])@V1t

        # while Q_q^T Q_q != U_q^T L U_q, their determinants are the same
        if self.log_det_S11 is None:
            self.log_det_S11 = self.logdet(np.swapaxes(Q[:,:self.q,:],-2,-1)@Q[:,:self.q,:])
        
        v = 1/np.atleast_1d(self.gamma)[:,None]*(self.p - np.linalg.norm(np.swapaxes(Q,-2,-1)[None,:,:,:]@M_tilde[:,None,:,:]@D_sqrtinv[:,None],axis=(-2,-1))**2)
        log_pdf = self.log_norm_constant - (self.q/2)*np.atleast_1d(log_det_D)[:,None] + (self.q-self.p-1)/2*self.log_det_S11[None] - 1/2*v

        return log_pdf

    def log_pdf_fullrank(self, Q):
        log_pdf = self.log_norm_constant - (self.q/2)*np.atleast_1d(self.logdet(self.Lambda))[:,None] - 1/2*np.trace(np.swapaxes(Q,-2,-1)[None,:]@np.linalg.inv(self.Lambda)[:,None]@Q[None,:],axis1=-2,axis2=-1)
        return log_pdf

    def log_pdf(self, Q):
        if self.distribution == 'SingularWishart_lowrank':
            return self.log_pdf_lowrank(Q)
        elif self.distribution == 'SingularWishart_fullrank':
            return self.log_pdf_fullrank(Q)
        
    def M_step_single_component(self,Q,beta,M=None,gamma=None,max_iter=int(1e8),tol=1e-10):
            
        if self.distribution == 'SingularWishart_lowrank':

            loss = []
            
            if gamma is None:
                raise ValueError("gamma is not provided")
            
            M_tilde = M * np.sqrt(1/gamma)
            _,S1,V1t = np.linalg.svd(M_tilde)
            D_inv = V1t.T@np.diag(1/(1+S1**2))@V1t

            trZt_oldZ_old = gamma**2*(np.sum(S1**4)+2*np.sum(S1**2)+self.p)
            gamma_old = gamma
            S1_old = S1
            M_tilde_old = M_tilde

            #precompute
            beta_sum = np.sum(beta)
            beta_S = np.sum(beta[:,None,None]*Q@np.swapaxes(Q,-2,-1),axis=0)

            for j in range(max_iter):

                # M_tilde update
                # M_tilde = 1/(gamma*self.q*beta_sum)*np.sum(beta_Q*QtM_tilde[:,None,:,:],axis=(0,-2))@D_inv
                M_tilde = 1/(gamma*self.q*beta_sum)*beta_S@M_tilde@D_inv
                # _,S1,V1t = np.linalg.svd(M_tilde)
                # D_inv = V1t.T@np.diag(1/(1+S1**2))@V1t
                S1 = np.linalg.svd(M_tilde,full_matrices=False,compute_uv=False)
                D_inv = np.linalg.inv(M_tilde.conj().T@M_tilde+np.eye(self.r))

                #gamma update
                # QtM_tilde = np.swapaxes(Q,-2,-1)@M_tilde
                # gamma = 1/(self.q*self.p*beta_sum)*np.sum(beta*(self.p-np.linalg.norm(QtM_tilde@V1t.T@np.diag(np.sqrt(1/(1+S1**2)))@V1t,axis=(-2,-1))**2))
                gamma = 1/(self.q*self.p*beta_sum)*(np.sum(beta*self.p)-np.trace(M_tilde@D_inv@M_tilde.T@beta_S))

                # convergence criterion
                trZtZ = gamma**2*(np.sum(S1**4)+2*np.sum(S1**2)+self.p)
                trZtZt_old = gamma*gamma_old*(np.linalg.norm(M_tilde.T@M_tilde_old)**2 + np.sum(S1**2) + np.sum(S1_old**2)+self.p)
                loss.append(trZtZ+trZt_oldZ_old-2*trZtZt_old)
                
                if j>0:
                    if loss[-1]<tol:
                        if np.any(np.array(loss)<-tol):
                            raise Warning("Loss is negative. Check M_step_single_component")
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
        elif self.distribution == 'SingularWishart_fullrank':
            Psi = 1/(self.q*np.sum(beta))*np.sum((beta[:,None,None]*Q)@np.swapaxes(Q,-2,-1),axis=0)
            return Psi

    def M_step(self,X):
        if self.K!=1:
            beta = np.exp(self.log_density-self.logsum_density)
            self.update_pi(beta)
        else:
            beta = np.ones((1,X.shape[0]))
        for k in range(self.K):
            if self.distribution == 'SingularWishart_lowrank':
                self.M[k],self.gamma[k] = self.M_step_single_component(X,beta[k],self.M[k],gamma=self.gamma[k])
            elif self.distribution == 'SingularWishart_fullrank':
                self.Lambda[k] = self.M_step_single_component(X,beta[k])