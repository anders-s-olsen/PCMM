import numpy as np
from scipy.special import loggamma
from src.load_HCP_data import initialize_pi_mu_M

class ACG():
    """
    Mixture model class
    """
    def __init__(self, K: int, p: int,rank=None,params=None):
        super().__init__()
        self.K = K
        self.p = p
        self.half_p = self.p/2
        self.r = rank
        # self.Z = np.zeros((self.K,self.p,self.p))
        self.logSA = loggamma(self.half_p) - np.log(2) -self.half_p* np.log(np.pi)

        if params is not None: # for evaluating likelihood with already-learned parameters
            self.pi = np.array(params['pi'])
            M_init = params['M']
            if M_init.dim()!=3 or M_init.shape[2]!=self.r: # add extra columns
                if M_init.dim()==2:
                    num_missing = self.r-1
                else:
                    num_missing = self.r-M_init.shape[2]

                M_extra = np.random.uniform(size=(self.K,self.p,num_missing))
                self.M = np.concatenate([M_init,M_extra],axis=2)
            else: 
                self.M = M_init

    def get_params(self):
        return {'M': self.M,'pi':self.pi,'c':self.c}
    
    def initialize(self,X=None,init=None,tol=None):
        self.pi,_,self.M = initialize_pi_mu_M(init=init,K=self.K,p=self.p,X=X,tol=tol,r=self.r,init_M=True)
        
################ E-step ###################
    def logdet(self,B):
        logdetsign,logdet = np.linalg.slogdet(B)
        return logdetsign*logdet

    def log_pdf(self,X):
        D = np.eye(self.r)+np.swapaxes(self.M,-2,-1)@self.M
        XM = X[None,:,:]@self.M
        v = 1-np.sum(XM@np.linalg.inv(D)*XM,axis=2) #check

        log_acg_pdf = self.logSA - 0.5*(self.logdet(D))[:,None] - self.half_p * np.log(v)
        return log_acg_pdf

    def log_density(self,X):
        return self.log_pdf(X)+np.log(self.pi)[:,None]
    
    def log_likelihood(self,X):
        self.density = self.log_density(X)
        self.logsum_density = np.logaddexp.reduce(self.density)
        loglik = np.sum(self.logsum_density)
        return loglik
    
    def posterior(self,X):
        density = self.log_density(X)
        logsum_density = np.logaddexp.reduce(density)
        return np.exp(density-logsum_density)
    
    def M_MLE_lowrank(self,M,X,weights = None,tol=1e-10,max_iter=10000):
        n,p = X.shape
        if n<p*(p-1):
            Warning("Too high dimensionality compared to number of observations. Lambda cannot be estimated")
        if weights is None:
            weights = np.ones(n)
        Q = weights[:,None]*X

        loss = []
        

        o = np.linalg.norm(M,'fro')**2
        b = 1/(1+o/p)
        M = np.sqrt(b)*M
        
        trMMtMMt_old = np.trace(M.T@M@M.T@M)
        M_old = M
        o_all = [o]

        for j in range(max_iter):

            # Woodbury scaled
            D_inv = np.linalg.inv(np.eye(self.r)+M.T@M)
            XM = X@M
            XMD_inv = XM@D_inv
            v = 1-np.sum(XMD_inv*XM,axis=1) #denominator
            M = p/np.sum(weights)*Q.T/v@XMD_inv
            o = np.linalg.norm(M,'fro')**2
            b = 1/(1+o/p)
            M = np.sqrt(b)*M

            trMMtMMt = np.trace(M.T@M@M.T@M)

            #Svarende til loss.append(np.linalg.norm(Z_old-Z)**2)
            # Kan man virkelig ikke reducere np.trace(M.T@M@M.T@M)??
            loss.append(trMMtMMt+trMMtMMt_old-2*np.trace(M@M.T@M_old@M_old.T))
            
            if j>0:
                if loss[-1]<tol:
                    break
            
            trMMtMMt_old = trMMtMMt
            M_old = M
            o_all.append(o)

        return M
    

    

############# M-step #################
    def M_step(self,X,tol=1e-10):
        n,_ = X.shape
        Beta = np.exp(self.density-self.logsum_density).T
        self.pi = np.sum(Beta,axis=0)/n

        for k in range(self.K):
            self.M[k] = self.M_MLE_lowrank(self.M[k],X,weights=Beta[:,k],tol=tol)
            
