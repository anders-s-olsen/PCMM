import numpy as np
from scipy.special import loggamma
from src.helper_functions2 import initialize_pi_mu_M

class ACG():
    """
    Mixture model class
    """
    def __init__(self, K: int, p: int,params=None):
        super().__init__()
        self.K = K
        self.p = p
        self.half_p = self.p/2
        self.logSA = loggamma(self.half_p) - np.log(2) -self.half_p* np.log(np.pi)

        if params is not None: # for evaluating likelihood with already-learned parameters
            self.Lambda = np.array(params['Lambda'])
            self.pi = np.array(params['pi'])

    def get_params(self):
        return {'Lambda': self.Lambda,'pi':self.pi}
    
    def initialize(self,X=None,init=None,tol=None):
        self.pi,mu,_ = initialize_pi_mu_M(init=init,K=self.K,p=self.p,X=X) 

        self.Lambda = np.zeros((self.K,self.p,self.p))    
        for k in range(self.K):
            self.Lambda[k] = 10e6*np.outer(mu[:,k],mu[:,k])+np.eye(self.p)
        
################ E-step ###################
    
    def log_norm_constant(self):
        logdetsign,logdet = np.linalg.slogdet(self.Lambda)
        return self.logSA - 0.5*logdetsign*logdet

    def log_pdf(self,X):
        pdf = np.sum(X@np.linalg.inv(self.Lambda)*X,axis=2)
        return self.log_norm_constant()[:,None] -self.half_p*np.log(pdf)

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

    def Lambda_MLE(self,Lambda,X,weights = None,tol=1e-10,max_iter=10000):
        n,p = X.shape
        if n<p*(p-1):
            print("Too high dimensionality compared to number of observations. Lambda cannot be calculated")
            return
        if weights is None:
            weights = np.ones(n)
        Lambda_old = np.eye(self.p)
        Q = weights[:,None]*X

        j = 0
        a = []
        while np.linalg.norm(Lambda_old-Lambda) > tol and (j < max_iter):
            # plt.figure(),plt.imshow(Lambda),plt.colorbar(),
            Lambda_old = Lambda
            
            # The following has been tested against a for-loop implementing Tyler1987 (3)
            # and also an implementation of Tyler1987 (2) where (2) has been evaluated and
            # Lambda=p/trace(Lambda)*Lambda for each iteration. All give the same result, 
            # at least before implementing weights

            XtLX = np.sum(X@np.linalg.inv(Lambda)*X,axis=1)
            # Lambda3 same as weights/XtLX
            Lambda = p/np.sum(weights/XtLX)*(Q.T/XtLX)@X

            j +=1
            a.append(np.linalg.norm(Lambda_old-Lambda))
        return Lambda
    

############# M-step #################
    def M_step(self,X,tol=1e-10):
        n,_ = X.shape
        Beta = np.exp(self.density-self.logsum_density).T
        self.pi = np.sum(Beta,axis=0)/n

        for k in range(self.K):
            self.Lambda[k] = self.Lambda_MLE(self.Lambda[k],X,weights=Beta[:,k],tol=tol)
