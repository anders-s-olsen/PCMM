import numpy as np
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
            self.distribution = 'ComplexSingularWishart_fullrank'
        else: #the lowrank version can also be full rank
            self.r = rank
            self.distribution = 'ComplexSingularWishart_lowrank'
        
        self.log_norm_constant = -self.p*self.q*np.log(np.pi)

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)
    
    def log_pdf_lowrank(self, X,L):
        # do the complex hermitian of M
        Mt = np.zeros_like(np.swapaxes(self.M,-2,-1),dtype=complex)
        for i in range(self.M.shape[0]):
            Mt[i] = self.M[i].H
        D = np.eye(self.r) + Mt@self.M 
        # Q = X*np.sqrt(L)[:,None,:]
        Qt = np.zeros_like(np.swapaxes(X,-2,-1),dtype=complex)
        for i in range(X.shape[0]):
            Qt[i] = X[i].H*np.sqrt(L[i])[:,None,:]
        QtM = Qt[:,None,:,:]@self.M[None,:,:,:]

        v = np.sum(L,axis=(-1))-np.trace(QtM@np.linalg.inv(D)[None]@np.swapaxes(QtM,-2,-1))

        log_pdf = self.log_norm_constant - (self.q/2)*self.logdet(D)[:,None] - 1/2*v 
        return log_pdf

    def log_pdf_fullrank(self, X,L):
        Qt = np.zeros_like(np.swapaxes(X,-2,-1),dtype=complex)
        for i in range(X.shape[0]):
            Qt[i] = X[i].H*np.sqrt(L[i])[:,None,:]
        Q = X*np.sqrt(L)[:,None,:]
        log_pdf = self.log_norm_constant - (self.q)*self.logdet(self.Lambda)[:,None] - np.trace(Qt[None,:]@np.linalg.inv(self.Lambda)[:,None]@Q[None,:],axis1=-2,axis2=-1)
        return log_pdf

    def log_pdf(self, X,L):
        if self.distribution == 'SingularWishart_lowrank':
            return self.log_pdf_lowrank(X,L)
        elif self.distribution == 'SingularWishart_fullrank':
            return self.log_pdf_fullrank(X,L)
        
    def M_step_single_component(self,X,L,Beta,M=None,max_iter=int(1e5),tol=1e-6):
        n,p,q = X.shape
        # if n<(p*(p-1)*q):
        #     Warning("Too high dimensionality compared to number of observations. Sigma cannot be estimated")
            
        if self.distribution == 'SingularWishart_lowrank':
            loss = []
            M_old = M.copy()
            o = np.linalg.norm(M,'fro')**2
            b = 1/(1+o/p)
            M = np.sqrt(b)*M
            Q = X*np.sqrt(L[:,None,:])
            # Q = Beta[:,None,None]*X
            trMMtMMt_old = np.linalg.norm(M)**2

            for j in range(max_iter):
                MtM = M.T@M

                # Woodbury scaled. First we update M
                D_inv = np.linalg.inv(np.eye(self.r)+MtM)
                QtM = np.swapaxes(Q,-2,-1)@M
                M = 1/(q*np.sum(Beta))*np.sum((Beta[:,None,None]*Q)@QtM,axis=0)@(np.eye(p)-D_inv@MtM)

                o = np.linalg.norm(M,'fro')**2
                b = 1/(1+o/p)
                M = np.sqrt(b)*M

                # To measure convergence, we compute norm(Z-Z_old)**2
                MtM = M.T@M
                trMMtMMt = np.linalg.norm(MtM)**2
                loss.append(trMMtMMt+trMMtMMt_old-2*np.linalg.norm(M.T@M_old)**2)
                
                if j>0:
                    if loss[-1]<tol:
                        break
                
                # To measure convergence
                trMMtMMt_old = trMMtMMt
                M_old = M
            return M
        elif self.distribution == 'SingularWishart_fullrank':
            Q = X*np.sqrt(L)[:,None,:]
            Psi = 1/(q*np.sum(Beta))*np.sum((Beta[:,None,None]*Q)@np.swapaxes(Q,-2,-1),axis=0)
            return Psi

    def M_step(self,X,L):
        Beta = np.exp(self.log_density-self.logsum_density)
        self.update_pi(Beta)
        for k in range(self.K):
            if self.distribution == 'SingularWishart_lowrank':
                self.M[k] = self.M_step_single_component(X,L,Beta[k],self.M[k])
            elif self.distribution == 'SingularWishart_fullrank':
                self.Lambda[k] = self.M_step_single_component(X,L,Beta[k],None)