import numpy as np
from scipy.special import loggamma
from src.helper_functions2 import initialize_pi_mu_M

class MACG():
    """
    Mixture model class
    """
    def __init__(self, K: int, p: int, q: int,params=None):
        super().__init__()
        self.K = K
        self.q = q
        self.p = p
        self.half_p = self.p/2
        loggamma_k = (self.q*(self.q-1)/4)*np.log(np.pi)+np.sum(loggamma(self.half_p-np.arange(self.q)/2))
        self.logSA_Stiefel = loggamma_k-self.q*np.log(2)-self.q*self.half_p*np.log(np.pi)

        if params is not None: # for evaluating likelihood with already-learned parameters
            self.Sigma = np.array(params['Sigma'])
            self.pi = np.array(params['pi'])

    def get_params(self):
        return {'Sigma': self.Sigma,'pi':self.pi}
    
    def initialize(self,X=None,init=None,tol=None):
        self.pi,mu,_ = initialize_pi_mu_M(init=init,K=self.K,p=self.p,X=X) 

        self.Sigma = np.zeros((self.K,self.p,self.p))    
        for k in range(self.K):
            self.Sigma[k] = 10e6*np.outer(mu[:,k],mu[:,k])+np.eye(self.p)
    
################ E-step ###################
    def logdet(self,B):
        logdetsign,logdet = np.linalg.slogdet(B)
        return logdetsign*logdet
    
    def log_pdf(self,X):
        pdf = self.logdet(np.swapaxes(X,-2,-1)[None,:,:,:]@np.linalg.inv(self.Sigma)[:,None,:,:]@X)
        return self.logSA_Stiefel - (self.q/2)*self.logdet(self.Sigma)[:,None] - self.half_p*pdf

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

    def Sigma_MLE(self,Sigma,X,weights = None,tol=1e-10,max_iter=10000):
        n,p,q = X.shape
        if n<(p*(p-1)*q):
            print("Too high dimensionality compared to number of observations. Sigma cannot be calculated")
            return
        if weights is None:
            weights = np.ones(n)
        Sigma_old = np.eye(self.p)
        Q = weights[:,None,None]*X
        
        j = 0
        while np.linalg.norm(Sigma_old-Sigma) > tol and (j < max_iter):
            Sigma_old = Sigma
            
            # this has been tested in the "Naive" version below
            XtLX = np.swapaxes(X,-2,-1)@np.linalg.inv(Sigma)@X
            L,V = np.linalg.eigh(XtLX)
            XtLX_trace = np.sum(1/L,axis=1) #trace of inverse is sum of inverse eigenvalues
            
            Sigma = p*np.sum(Q@np.linalg.inv(XtLX)@np.swapaxes(X,-2,-1),axis=0) \
                /(np.sum(weights*XtLX_trace))
            
            # Ldiag = 1/L[..., np.newaxis] * np.eye(self.q)
            # Sigma = p*np.sum(Q@V@Ldiag@np.swapaxes(V,-2,-1)@np.swapaxes(X,-2,-1),axis=0) \
            #     /(np.sum(weights*XtLX_trace))
            j +=1
        return Sigma
    

############# M-step #################
    def M_step(self,X,tol=1e-10):
        n,q,p = X.shape
        Beta = np.exp(self.density-self.logsum_density).T
        self.pi = np.sum(Beta,axis=0)/n

        for k in range(self.K):
            self.Sigma[k] = self.Sigma_MLE(self.Sigma[k],X,weights=Beta[:,k],tol=tol)
