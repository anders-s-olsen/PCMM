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
    def __init__(self, p:int, K:int=1, complex:bool=False, params:dict=None):
        super().__init__()

        self.K = K
        self.p = p
        
        if complex:
            self.distribution = 'Complex_Watson'
            self.c = self.p-1
            self.a = 1
        else:
            self.distribution = 'Watson'
            self.c = self.p/2
            self.a = 0.5

        self.logSA_sphere = loggamma(self.c) - np.log(2) - self.c* np.log(np.pi)

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)
    
    def log_norm_constant(self):
        logkum = np.zeros(self.K)
        for idx,kappa in enumerate(self.kappa):
            logkum[idx] = kummer_log(k=kappa,a=self.a,c=self.c)
        return self.logSA_sphere - logkum
    
    def log_pdf(self, X):
        #the abs added for support of complex arrays
        log_pdf = self.log_norm_constant()[:,None] + self.kappa[:,None]*(np.abs(self.mu.H@X.H)**2)
        return log_pdf
    
    def M_step_single_component(self,X,beta,mu,kappa,tol=1e-8):
        n,p = X.shape
        Q = np.sqrt(beta)[:,None]*X

        # the folllowing options are optimized for n>p but should work otherwise
        if n>p:
            if kappa>0:
                _,_,mu = svds(Q,k=1,which='LM',v0=mu,return_singular_vectors='vh')
            elif kappa<0:
                _,_,mu = svds(Q,k=1,which='SM',v0=mu,return_singular_vectors='vh')
        else: #no start vector because? svds throws error
            if kappa>0:
                _,_,mu = svds(Q,k=1,which='LM',return_singular_vectors='vh')
            elif kappa<0:
                _,_,mu = svds(Q,k=1,which='SM',return_singular_vectors='vh')

        rk = 1/np.sum(beta)*np.sum((mu@Q.H)**2)
        LB = (rk*self.c-self.a)/(rk*(1-rk))*(1+(1-rk)/(self.c-self.a))
        B  = (rk*self.c-self.a)/(2*rk*(1-rk))*(1+np.sqrt(1+4*(self.c+1)*rk*(1-rk)/(self.a*(self.c-self.a))))
        UB = (rk*self.c-self.a)/(rk*(1-rk))*(1+rk/self.a)

        def f(kappa):
            return ((self.a/self.c)*(np.exp(kummer_log(k=kappa,a=self.a+1,c=self.c+1)-kummer_log(k=kappa,a=self.a,c=self.c)))-rk)**2
        options={}
        options['xatol'] = tol
        if rk>self.a/self.c:
            kappa = minimize_scalar(f, bounds=[LB, B],options=options,method='bounded')['x']
            if kappa<LB or kappa>B:
                print('Probably a convergence problem for kappa')
                return
        elif rk<self.a/self.c:
            kappa = minimize_scalar(f, bounds=[B, UB],options=options,method='bounded')['x']
            if kappa<B or kappa>UB:
                print('Probably a convergence problem for kappa')
                return
        elif rk==self.a/self.c:
            kappa = 0
        else:
            raise ValueError("kappa could not be optimized")
        return mu,kappa
    
    def M_step(self,X):
        # n,p = X.shape
        if self.K!=1:
            beta = np.exp(self.log_density-self.logsum_density)
            self.update_pi(beta)
        else:
            beta = np.ones((1,X.shape[0]))

        for k in range(self.K):
            self.mu[:,k],self.kappa[k] = self.M_step_single_component(X=X,beta=beta[k],mu=self.mu[:,k],kappa=self.kappa[k])
            
if __name__ == '__main__':
    import scipy

    #unit test in 2d
    W = Watson(p=2,K=1,params={'mu':np.array([[1,0]]).T,'kappa':np.array([1.])})

    W_pdf = lambda phi: float(np.exp(W.log_pdf(np.array([np.cos(phi), np.sin(phi)]))))
    w_result2d = scipy.integrate.quad(W_pdf, 0., 2*np.pi)[0]

    # unit test in 3d
    W = Watson(p=3,K=1,params={'mu':np.array([[1,0,0]]).T,'kappa':np.array([1.])})

    W_pdf = lambda theta,phi: float(np.exp(W.log_pdf(np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]))))
    w_result3d = scipy.integrate.dblquad(W_pdf, 0., np.pi, 0., 2*np.pi)[0]

    print(w_result2d,w_result3d)
