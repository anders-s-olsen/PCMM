import numpy as np
from scipy.special import loggamma
import scipy
from src.DMM_EM.DMMEMBaseModel import DMMEMBaseModel

class SingularWishart(DMMEMBaseModel):
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
        self.log_norm_constant = self.q**2/2*np.log(np.pi)+loggamma_k-self.q*self.half_p*np.log(2*np.pi)
        self.log_det_S11 = None

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)
    
    def log_pdf_lowrank(self, Q):
        # Q defined as X*np.sqrt(L)[:,None,:]
        QtM = np.swapaxes(Q,-2,-1)[None,:,:,:]@self.M[:,None,:,:]

        _,S1,V1t = np.linalg.svd(self.M)
        D_sqrtinv = (np.swapaxes(V1t,-2,-1)*np.sqrt(1/(1+S1**2))[:,None])@V1t
        log_det_D = np.sum(np.log(1/S1**2+1),axis=-1)+2*np.sum(np.log(S1),axis=-1)

        # while Q_q^T Q_q != U_q^T L U_q, their determinants are the same'
        if self.log_det_S11 is None:
            self.log_det_S11 = self.logdet(np.swapaxes(Q[:,:self.q,:],-2,-1)@Q[:,:self.q,:])

        v = self.p - np.linalg.norm(QtM@D_sqrtinv[:,None],axis=(-2,-1))**2
        log_pdf = self.log_norm_constant - (self.q/2)*log_det_D[:,None] + (self.q-self.p-1)/2*self.log_det_S11[None] - 1/2*v

        # QtM = np.swapaxes(Q,-2,-1)[:,None,:,:]@self.M[None,:,:,:]
        # D = np.eye(self.r) + np.swapaxes(self.M,-2,-1)@self.M 
        
        # D_sqrtinv = np.zeros((self.K,self.r,self.r))
        # for k in range(self.K):
        #     D_sqrtinv[k] = scipy.linalg.sqrtm(np.linalg.inv(D[k]))

        # # v = self.p-np.trace(QtM@np.linalg.inv(D)[None]@np.swapaxes(QtM,-2,-1),axis1=-2,axis2=-1)
        # v = self.p - np.linalg.norm(QtM@D_sqrtinv,axis=(-2,-1))**2

        # log_pdf = self.log_norm_constant - (self.q/2)*self.logdet(D)[:,None] - 1/2*v.T
        return log_pdf

    def log_pdf_fullrank(self, Q):
        log_pdf = self.log_norm_constant - (self.q/2)*self.logdet(self.Lambda)[:,None] - 1/2*np.trace(np.swapaxes(Q,-2,-1)[None,:]@np.linalg.inv(self.Lambda)[:,None]@Q[None,:],axis1=-2,axis2=-1)
        return log_pdf

    def log_pdf(self, Q):
        if self.distribution == 'SingularWishart_lowrank':
            return self.log_pdf_lowrank(Q)
        elif self.distribution == 'SingularWishart_fullrank':
            return self.log_pdf_fullrank(Q)
        
    def M_step_single_component(self,Q,Beta,M=None,max_iter=int(1e5),tol=1e-6):
        # n,p,q = Q.shape
            
        if self.distribution == 'SingularWishart_lowrank':

            loss = []
            M_old = M.copy()
            _,S1,V1t = np.linalg.svd(M)
            trMMtMMt_old = np.sum(S1**4) #+2*np.sum(S1**2)+p cancels out

            for j in range(max_iter):
                # # if j divisible by 100 print j and the loss
                # if j>0:
                #     if j%100==0:
                #         print(j,loss[-1])

                # Woodbury scaled. First we update M
                QtM = np.swapaxes(Q,-2,-1)@M
                D_inv =V1t.T@np.diag(1/(1+S1**2))@V1t
                M = 1/(self.q*np.sum(Beta))*np.sum((Beta[:,None,None]*Q)[:,:,:,None]*QtM[:,None,:,:],axis=(0,-2))@D_inv


                # To measure convergence, we compute norm(Z-Z_old)**2
                _,S1,V1t = np.linalg.svd(M)
                trMMtMMt = np.sum(S1**4) #+2*np.sum(S1**2)+p
                loss.append(trMMtMMt+trMMtMMt_old-2*np.linalg.norm(M.T@M_old)**2)
                
                if j>0:
                    if loss[-1]<tol:
                        if np.any(np.array(loss)<0):
                            raise Warning("Loss is negative. Check M_step_single_component")
                        break
                
                # To measure convergence
                trMMtMMt_old = trMMtMMt
                M_old = M
            return M
        elif self.distribution == 'SingularWishart_fullrank':
            # Q = X*np.sqrt(L)[:,None,:]
            Psi = 1/(self.q*np.sum(Beta))*np.sum((Beta[:,None,None]*Q)@np.swapaxes(Q,-2,-1),axis=0)
            return Psi

    def M_step(self,X):
        Beta = np.exp(self.log_density-self.logsum_density)
        self.update_pi(Beta)
        for k in range(self.K):
            if self.distribution == 'SingularWishart_lowrank':
                self.M[k] = self.M_step_single_component(X,Beta[k],self.M[k])
            elif self.distribution == 'SingularWishart_fullrank':
                self.Lambda[k] = self.M_step_single_component(X,Beta[k],None)