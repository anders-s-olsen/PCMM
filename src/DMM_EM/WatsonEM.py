import numpy as np
from src.DMM_EM.DMMEMBaseModel import DMMEMBaseModel
from scipy.sparse.linalg import svds
from scipy.optimize import minimize_scalar
from scipy.special import loggamma
# import numba as nb

# @nb.njit(nb.float64(nb.float64,nb.float64,nb.float64),cache=True)#
def kummer_log(k,a,c):
    n = 1e7
    tol = 1e-10
    logkum = 0
    logkum_old = 1
    foo = 0
    if k<0:
        a = c-a
    j = 1
    while np.abs(logkum - logkum_old) > tol and (j < n):
        logkum_old = logkum
        foo += np.log((a + j - 1) / (j * (c + j - 1)) * np.abs(k))
        logkum = np.logaddexp(logkum,foo)
        j += 1
    return logkum

class Watson(DMMEMBaseModel):
    def __init__(self, p:int, K:int=1, params:dict=None):
        super().__init__()

        self.K = K
        self.p = p
        self.half_p = self.p/2
        self.a = 0.5
        self.distribution = 'Watson'
        
        # precompute log-surface area of the unit hypersphere
        self.logSA_sphere = loggamma(self.half_p) - np.log(2) -self.half_p* np.log(np.pi)

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)
    
    def log_norm_constant(self):
        logkum = np.zeros(self.K)
        for idx,kappa in enumerate(self.kappa):
            logkum[idx] = kummer_log(k=kappa,a=self.a,c=self.half_p)
        return self.logSA_sphere - logkum
    
    def log_pdf(self, X):
        log_pdf = self.log_norm_constant()[:,None] + self.kappa[:,None]*((self.mu.T@X.T)**2)
        return log_pdf
    
    def M_step(self,X):
        n,p = X.shape
        Beta = np.exp(self.log_density-self.logsum_density)
        self.update_pi(Beta)
        # self.pi = np.sum(Beta,axis=0)/n

        for k in range(self.K):
            Q = np.sqrt(Beta[k])[:,None]*X

            # the folllowing options are optimized for n>p but should work otherwise
            if self.kappa[k]>0:
                _,_,self.mu[:,k] = svds(Q,k=1,which='LM',v0=self.mu[:,k],return_singular_vectors='vh')
            elif self.kappa[k]<0:
                _,_,self.mu[:,k] = svds(Q,k=1,which='SM',v0=self.mu[:,k],return_singular_vectors='vh')

            rk = 1/np.sum(Beta[k])*np.sum((self.mu[:,k].T@Q.T)**2)
            LB = (rk*self.half_p-self.a)/(rk*(1-rk))*(1+(1-rk)/(self.half_p-self.a))
            B  = (rk*self.half_p-self.a)/(2*rk*(1-rk))*(1+np.sqrt(1+4*(self.half_p+1)*rk*(1-rk)/(self.a*(self.half_p-self.a))))
            UB = (rk*self.half_p-self.a)/(rk*(1-rk))*(1+rk/self.a)
            # BBG = (self.half_p*rk-self.a)/(rk*(1-rk))+rk/(2*self.half_p*(1-rk))

            def f(kappa):
                return ((self.a/self.half_p)*(np.exp(kummer_log(k=kappa,a=self.a+1,c=self.half_p+1)-kummer_log(k=kappa,a=self.a,c=self.half_p)))-rk)**2

            if rk>self.a/self.half_p:
                self.kappa[k] = minimize_scalar(f, bounds=[LB, B],tol=None,method='bounded')['x']
                if self.kappa[k]<LB or self.kappa[k]>B:
                    print('Probably a convergence problem for kappa')
                    return
            elif rk<self.a/self.half_p:
                self.kappa[k] = minimize_scalar(f, bounds=[B, UB],tol=None,method='bounded')['x']
                if self.kappa[k]<B or self.kappa[k]>UB:
                    print('Probably a convergence problem for kappa')
                    return
            elif rk==self.a/self.half_p:
                self.kappa[k] = 0
            else:
                raise ValueError("kappa could not be optimized")
            # if np.linalg.norm(self.kappa[k]-LB)<1e-10 or np.linalg.norm(self.kappa[k]-B)<1e-10 or np.linalg.norm(self.kappa[k]-UB)<1e-10:
            #     print('Probably a convergence problem for kappa')
            #     return
