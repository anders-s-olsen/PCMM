import numpy as np
from src.DMM_EM.DMMEMBaseModel import DMMEMBaseModel
from scipy.special import loggamma

class ACG(DMMEMBaseModel):
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
        if self.distribution in ['ACG_lowrank','Complex_ACG_lowrank']:
            return self.log_pdf_lowrank(X)
        elif self.distribution in ['ACG_fullrank','Complex_ACG_fullrank']:
            return self.log_pdf_fullrank(X)

    # def M_step_more_components(self,X,beta,M=None,Lambda=None,max_iter=int(1e5),tol=1e-8):
    #     if self.distribution == 'ACG_lowrank':
    #         Q = np.swapaxes(beta[:,:,None]*X[None,:,:],-2,-1)
            
    #         loss = []
    #         for k in range(self.K):
    #             loss.append([])
    #         M_out = np.zeros_like(M)
    #         gamma_out = np.zeros(self.K)
    #         Ks = np.arange(self.K)
            
    #         o = np.linalg.norm(M,'fro',axis=(-2,-1))**2
    #         gamma = 1/(1+o/self.p)
    #         M = np.sqrt(gamma)[:,None,None]*M 
    #         _,S1,V1t = np.linalg.svd(M,full_matrices=False)

    #         trZt_oldZ_old = np.sum(S1**4,axis=-1)+2*gamma*np.sum(S1**2,axis=-1)+gamma**2*self.p
    #         gamma_old = gamma
    #         S1_old = S1
    #         M_old = M.copy()

    #         for j in range(max_iter):

    #             # First we update M
    #             D_inv = (np.swapaxes(V1t,-2,-1)*(1/(1+S1**2))[:,None])@V1t
    #             XM = X[None,:,:]@M
    #             XMD_inv = XM@D_inv
    #             v = 1-np.sum(XMD_inv*XM,axis=-1) #denominator
    #             M = self.p/np.sum(beta,axis=-1)[:,None,None]*Q@(XMD_inv/v[:,:,None])

    #             # Then we trace-normalize M
    #             o = np.linalg.norm(M,'fro',axis=(-2,-1))**2
    #             gamma = 1/(1+o/self.p)
    #             M = np.sqrt(gamma)[:,None,None]*M

    #             # To measure convergence, we compute norm(Z-Z_old)**2
    #             _,S1,V1t = np.linalg.svd(M,full_matrices=False)
    #             trZtZ = np.sum(S1**4,axis=-1)+2*gamma*np.sum(S1**2,axis=-1)+gamma**2*self.p
    #             trZtZt_old = np.linalg.norm(np.swapaxes(M,-2,-1)@M_old,axis=(-2,-1))**2 + gamma_old*gamma*self.p + gamma_old*np.sum(S1**2,axis=-1) + gamma*np.sum(S1_old**2,axis=-1)
    #             for k in range(len(Ks)):
    #                 loss[Ks[k]].append(trZtZ[k]+trZt_oldZ_old[k]-2*trZtZt_old[k])

    #             if j>0:
    #                 if np.any(np.concatenate(loss)<-tol):
    #                     raise Warning("Loss is negative. Check M_step_more_components")
    #                 to_be_popped = []
    #                 for k in range(len(Ks)):
    #                     if loss[Ks[k]][-1]<tol:
    #                         to_be_popped.append(k)
                    
    #                 M_out[Ks[to_be_popped]] = M[to_be_popped]
    #                 gamma_out[Ks[to_be_popped]] = gamma[to_be_popped]
    #                 # pop out the element of M
    #                 Ks = np.delete(Ks,to_be_popped)
    #                 if len(Ks)==0:
    #                     break
    #                 M = np.delete(M,to_be_popped,axis=0)
    #                 V1t = np.delete(V1t,to_be_popped,axis=0)
    #                 S1 = np.delete(S1,to_be_popped,axis=0)
    #                 gamma = np.delete(gamma,to_be_popped)
    #                 trZtZ = np.delete(trZtZ,to_be_popped)
    #                 beta = np.delete(beta,to_be_popped,axis=0)
    #                 Q = np.delete(Q,to_be_popped,axis=0)

    #             # To measure convergence
    #             trZt_oldZ_old = trZtZ
    #             gamma_old = gamma
    #             M_old = M
    #             S1_old = S1

    #         if j==max_iter-1:
    #             raise Warning("M-step did not converge")
            
    #         # output the unnormalized M such that the log_pdf is computed without having to include the normalization factor (pdf is scale invariant)
    #         M_out = M_out/np.sqrt(gamma_out[:,None,None])
    #         return M_out
            
    def M_step_single_component(self,X,beta,M=None,Lambda=None,max_iter=int(1e5),tol=1e-10):
        n,p = X.shape
        if n<p*(p-1):
            Warning("Too high dimensionality compared to number of observations. Lambda cannot be calculated")
        if self.distribution in ['ACG_lowrank','Complex_ACG_lowrank']:
            Q = (beta[:,None]*X).T

            loss = []
            o = np.linalg.norm(M,'fro')**2
            gamma = 1/(1+o/self.p)
            M = np.sqrt(gamma)*M #we control the scale of M to avoid numerical issues
            _,S1,V1h = np.linalg.svd(M,full_matrices=False)
            
            trZt_oldZ_old = np.sum(S1**4)+2*gamma*np.sum(S1**2)+gamma**2*self.p
            gamma_old = gamma
            S1_old = S1
            M_old = M.copy()

            for j in range(max_iter):

                # First we update M
                D_inv = V1h.conj().T@np.diag(1/(1+S1**2))@V1h
                XM = X.conj()@M #check the conjs here and below
                XMD_inv = XM@D_inv
                v = 1-np.sum(XMD_inv*XM.conj(),axis=1) #denominator
                M = self.p/np.sum(beta)*Q@(XMD_inv/v[:,None])

                # Then we trace-normalize M
                o = np.linalg.norm(M,'fro')**2
                gamma = 1/(1+o/self.p)
                M = np.sqrt(gamma)*M

                # To measure convergence, we compute norm(Z-Z_old)**2
                _,S1,V1h = np.linalg.svd(M,full_matrices=False)
                trZtZ = np.sum(S1**4)+2*gamma*np.sum(S1**2)+gamma**2*self.p
                trZtZt_old = np.linalg.norm(M.conj().T@M_old)**2 + gamma_old*gamma*self.p + gamma_old*np.sum(S1**2) + gamma*np.sum(S1_old**2)
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
        elif self.distribution in ['ACG_fullrank','Complex_ACG_fullrank']:
            Lambda_old = np.eye(self.p)
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
        # if self.distribution == 'ACG_lowrank':
        #     self.M = self.M_step_more_components(X,beta,self.M)
        # elif self.distribution == 'ACG_fullrank':
        #     for k in range(self.K):
        #         self.Lambda[k] = self.M_step_single_component(X,beta[k],None,self.Lambda[k])


if __name__ == "__main__":
    import scipy
    # unit test
    ACG_model = ACG(p=2, K=1, rank=1,params={'M':np.array([[1,0]]).T})
    ACG_pdf = lambda phi: float(np.exp(ACG_model.log_pdf(np.array([[np.cos(phi), np.sin(phi)]]))))
    acg_result2d_lowrank = scipy.integrate.quad(ACG_pdf, 0., 2*np.pi)

    ACG_model = ACG(p=2, K=1, rank=None,params={'Lambda':np.array([[1,0]])@np.array([[1,0]]).T+np.eye(2)})
    ACG_pdf = lambda phi: float(np.exp(ACG_model.log_pdf(np.array([[np.cos(phi), np.sin(phi)]]))))
    acg_result2d_fullrank = scipy.integrate.quad(ACG_pdf, 0., 2*np.pi)

    # in 3d
    ACG_model = ACG(p=3, K=1, rank=1,params={'M':np.array([[1,0,0]]).T})
    ACG_pdf = lambda phi, theta: float(np.exp(ACG_model.log_pdf(np.array([[np.cos(theta)*np.cos(phi), np.sin(theta)*np.cos(phi), np.sin(phi)]]))))
    acg_result3d_lowrank = scipy.integrate.dblquad(ACG_pdf, 0., np.pi, 0., 2*np.pi)

    ACG_model = ACG(p=3, K=1, rank=None,params={'Lambda':np.array([[1,0,0]])@np.array([[1,0,0]]).T+np.eye(3)})
    ACG_pdf = lambda phi, theta: float(np.exp(ACG_model.log_pdf(np.array([[np.cos(theta)*np.cos(phi), np.sin(theta)*np.cos(phi), np.sin(phi)]]))))
    acg_result3d_fullrank = scipy.integrate.dblquad(ACG_pdf, 0., np.pi, 0., 2*np.pi)

    print(acg_result2d_lowrank[0], acg_result2d_fullrank[0], acg_result3d_lowrank[0], acg_result3d_fullrank[0])