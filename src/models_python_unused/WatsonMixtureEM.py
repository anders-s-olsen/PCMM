import numpy as np
from scipy.special import loggamma
import scipy
from src.load_HCP_data import initialize_pi_mu_M

class Watson():
    """
    Mixture model class
    """
    def __init__(self, K: int, p: int,params=None):
        super().__init__()
        self.K = K
        self.p = p
        self.c = self.p/2
        self.a = 0.5
        self.logSA = loggamma(self.c) - np.log(2) -self.c* np.log(np.pi)
        self.loglik = []

        if params is not None: # for evaluating likelihood with already-learned parameters
            self.mu = np.array(params['mu'])
            self.kappa = np.array(params['kappa'])
            self.pi = np.array(params['pi'])

    def get_params(self):
        return {'mu': self.mu,'kappa':self.kappa,'pi':self.pi}
    
    def initialize(self,X=None,init=None,tol=None):
        self.pi,self.mu,_ = initialize_pi_mu_M(init=init,K=self.K,p=self.p,X=X) 
        self.kappa = np.ones(self.K)
    

################ E-step ###################
    def kummer_log(self,a, c, k, n=1000000,tol=1e-10):
        logkum = np.zeros((k.size))
        logkum_old = np.ones((k.size))
        foo = np.zeros((k.size))
        j = 1
        while np.any(np.abs(logkum - logkum_old) > tol) and (j < n):
            logkum_old = logkum
            foo += np.log((a + j - 1) / (j * (c + j - 1)) * k)
            logkum = np.logaddexp(logkum,foo)
            j += 1
        return logkum    
    
    def log_norm_constant(self):
        # return self.logSA - np.log(hyp1f1(self.a,self.c,self.kappa)) 
        return self.logSA - self.kummer_log(self.a,self.c,self.kappa)[:,None]
        # return self.logSA - self.log_kummer_np(self.a,self.c,self.kappa)
    
    def log_pdf(self,X):
        return self.log_norm_constant() + self.kappa[:,None]*((self.mu.T@X.T)**2)

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

############# M-step #################
    def M_step(self,X,tol=1e-10):
        n,p = X.shape
        Beta = np.exp(self.density-self.logsum_density).T
        self.pi = np.sum(Beta,axis=0)/n

        for k in range(self.K):
            Q = np.sqrt(Beta[:,k])[:,None]*X

            # the folllowing options are optimized for n>p but should work otherwise
            if self.kappa[k]>0:
                _,_,self.mu[:,k] = scipy.sparse.linalg.svds(Q,k=1,which='LM',v0=self.mu[:,k],return_singular_vectors='vh')
                # [~,~,mu(:,k)]=svds(Q,1,'largest','RightStartVector',mu_old(:,k));
                # self.mu[:,k] = V[:,0]
            elif self.kappa[k]<0:
                _,_,self.mu[:,k] = scipy.sparse.linalg.svds(Q,k=1,which='SM',v0=self.mu[:,k],return_singular_vectors='vh')
                # self.mu[:,k] = V[:,-1]

            rk = 1/np.sum(Beta[:,k])*np.sum((self.mu[:,k].T@Q.T)**2)
            LB = (rk*self.c-self.a)/(rk*(1-rk))*(1+(1-rk)/(self.c-self.a))
            B  = (rk*self.c-self.a)/(2*rk*(1-rk))*(1+np.sqrt(1+4*(self.c+1)*rk*(1-rk)/(self.a*(self.c-self.a))))
            UB = (rk*self.c-self.a)/(rk*(1-rk))*(1+rk/self.a)
            BBG = (self.c*rk-self.a)/(rk*(1-rk))+rk/(2*self.c*(1-rk))

            # def f(kappa):
            #     return -(kappa * rk - hyp1f1(self.a, self.c, kappa))
            def f(kappa):
                return ((self.a/self.c)*(np.exp(self.kummer_log(self.a+1,self.c+1,kappa)-self.kummer_log(self.a,self.c,kappa)))-rk)**2

            if rk>self.a/self.c:
                # self.kappa[k] = scipy.optimize.minimize(f, x0=BBG, bounds=[(LB, B)],tol=None)['x']
                self.kappa[k] = scipy.optimize.minimize_scalar(f, bounds=[LB, B],tol=None)['x']
            elif rk<self.a/self.c:
                # self.kappa[k] = scipy.optimize.minimize(f, x0=BBG, bounds=[(B, UB)],tol=None)['x']
                self.kappa[k] = scipy.optimize.minimize_scalar(f, bounds=[B, UB],tol=None)['x']
            elif rk==self.a/self.c:
                self.kappa[k] = 0
            else:
                raise ValueError("kappa could not be optimized")
            if np.linalg.norm(self.kappa[k]-LB)<1e-10 or np.linalg.norm(self.kappa[k]-B)<1e-10 or np.linalg.norm(self.kappa[k]-UB)<1e-10:
                print('Probably a convergence problem for kappa')
                return