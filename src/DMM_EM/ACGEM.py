import numpy as np
from src.DMM_EM.DMMEMBaseModel import DMMEMBaseModel
from scipy.special import loggamma

class ACG(DMMEMBaseModel):
    def __init__(self, p:int, rank:int=None, K:int=1, params:dict=None):
        super().__init__()

        self.K = K
        self.p = p
        if rank is None or rank==0: #rank zero means the fullrank version
            self.r = p
            self.distribution = 'ACG_fullrank'
        else: #the lowrank version can also be full rank
            self.r = rank
            self.distribution = 'ACG_lowrank'
        
        # precompute log-surface area of the unit hypersphere
        self.logSA_sphere = loggamma(self.p/2) - np.log(2) -self.p/2* np.log(np.pi)

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)
    
    def log_pdf_lowrank(self, X):
        _,S1,V1t = np.linalg.svd(self.M,full_matrices=False)
        log_det_D = np.sum(np.log(1/S1**2+1),axis=-1)+2*np.sum(np.log(S1),axis=-1)
        D_inv = (np.swapaxes(V1t,-2,-1)*(1/(1+S1**2))[:,None])@V1t

        XM = X[None,:,:]@self.M
        v = 1-np.sum(XM@D_inv*XM,axis=-1) 

        log_pdf = self.logSA_sphere - 0.5 * np.atleast_1d(log_det_D)[:,None] - self.p/2 * np.log(v)
        return log_pdf
    
    def log_pdf_fullrank(self, X):
        XtLX = np.sum(X@np.linalg.inv(self.Lambda)*X,axis=-1)
        log_pdf = self.logSA_sphere - 0.5 * np.atleast_1d(self.logdet(self.Lambda))[:,None] -self.p/2*np.log(XtLX)
        return log_pdf

    def log_pdf(self, X,L=None):
        if self.distribution == 'ACG_lowrank':
            return self.log_pdf_lowrank(X)
        elif self.distribution == 'ACG_fullrank':
            return self.log_pdf_fullrank(X)

    def M_step_single_component(self,X,beta,M=None,Lambda=None,max_iter=int(1e5),tol=1e-8):
        n,p = X.shape
        if n<p*(p-1):
            Warning("Too high dimensionality compared to number of observations. Lambda cannot be calculated")
        if self.distribution == 'ACG_lowrank':
            Q = (beta[:,None]*X).T

            loss = []
            o = np.linalg.norm(M,'fro')**2
            gamma = 1/(1+o/p)
            M = np.sqrt(gamma)*M #we control the scale of M to avoid numerical issues
            _,S1,V1t = np.linalg.svd(M,full_matrices=False)
            # M_tilde = np.sqrt(gamma)*M #we control the scale of M to avoid numerical issues
            # _,S1,V1t = np.linalg.svd(M_tilde,full_matrices=False)
            
            trZt_oldZ_old = np.sum(S1**4)+2*gamma*np.sum(S1**2)+gamma**2*self.p
            gamma_old = gamma
            S1_old = S1
            M_old = M.copy()

            for j in range(max_iter):

                # First we update M
                D_inv = V1t.T@np.diag(1/(1+S1**2))@V1t
                XM = X@M
                XMD_inv = XM@D_inv
                v = 1-np.sum(XMD_inv*XM,axis=1) #denominator
                M = p/np.sum(beta)*Q@(XMD_inv/v[:,None])

                # Then we trace-normalize M
                o = np.linalg.norm(M,'fro')**2
                gamma = 1/(1+o/p)
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
        elif self.distribution == 'ACG_fullrank':
            Lambda_old = np.eye(self.p)
            Q = beta[:,None]*X
            for j in range(max_iter):
                XtLX = np.sum(X@np.linalg.inv(Lambda)*X,axis=1)
                Lambda = p/np.sum(beta/XtLX)*(Q.T/XtLX)@X
                if j>0:
                    if np.linalg.norm(Lambda_old-Lambda)<tol:
                        break
                Lambda_old = Lambda
            return Lambda

    def M_step(self,X):
        beta = np.exp(self.log_density-self.logsum_density)
        self.update_pi(beta)
        for k in range(self.K):
            if self.distribution == 'ACG_lowrank':
                self.M[k] = self.M_step_single_component(X,beta[k],self.M[k],None)
            elif self.distribution == 'ACG_fullrank':
                self.Lambda[k] = self.M_step_single_component(X,beta[k],None,self.Lambda[k])

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