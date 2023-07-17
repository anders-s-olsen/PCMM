import numpy as np
from scipy.special import loggamma, gamma
from src.models_python.diametrical_clustering import diametrical_clustering, diametrical_clustering_plusplus
from src.models_python.WatsonMixtureEM import Watson
from src.models_python.mixture_EM_loop import mixture_EM_loop

import time

class MACG():
    """
    Mixture model class
    """
    def __init__(self, K: int, p: int, q: int,params=None):
        super().__init__()
        self.K = K
        self.q = q
        self.p = p
        self.c = self.p/2
        self.gamma_k = np.pi**(self.q*(self.q-1)/4)
        for i in range(self.q):
            self.gamma_k *= gamma(self.c-i/2) # should be (i-1)/2 but python is zero-indexed :(
        self.logSA_Stiefel = np.log(self.gamma_k)-q*np.log(2)-self.q*self.p/2*np.log(np.pi)

        if params is not None: # for evaluating likelihood with already-learned parameters
            self.Sigma = np.array(params['Sigma'])
            self.pi = np.array(params['pi'])

        
    def get_params(self):
        return {'Sigma': self.Sigma,'pi':self.pi}
    
    def initialize(self,X=None,init=None,tol=None):
        self.pi = np.repeat(1/self.K,repeats=self.K)
        if init is None or init=='uniform' or init=='unif':
            mu = np.random.uniform(size=(self.p,self.K))
            mu = mu/np.linalg.norm(mu,axis=0)
        elif init == "test":
            mu = np.vstack((np.array([1,1,1]),np.array([1,1,-1]))).T
            mu = mu/np.linalg.norm(mu,axis=0)
        elif init == '++' or init == 'plusplus' or init == 'diametrical_clustering_plusplus':
            mu = diametrical_clustering_plusplus(X=X[:,:,0],K=self.K)
        elif init == 'dc' or init == 'diametrical_clustering':
            mu = diametrical_clustering(X=X[:,:,0],K=self.K,max_iter=100000,num_repl=5,init='++',tol=tol)
        elif init == 'WMM' or init == 'Watson' or init == 'W' or init == 'watson':
            W = Watson(K=self.K,p=self.p)
            params,_,_,_ = mixture_EM_loop(W,X[:,:,0],init='dc')
            mu = params['mu']
            self.pi = params['pi']
            
        self.Sigma = np.zeros((self.K,self.p,self.p))
        for k in range(self.K):
            self.Sigma[k] = np.outer(mu[:,k],mu[:,k])+np.eye(self.p)
    

################ E-step ###################
    
    def log_norm_constant(self):
        return self.logSA_Stiefel - (self.q/2)*np.log(np.linalg.det(self.Sigma))

    def log_pdf(self,X):
        pdf = np.zeros((self.K,X.shape[0]))
        for k in range(self.K):
            pdf[k] = np.linalg.det(np.swapaxes(X,-2,-1)@np.linalg.inv(self.Sigma[k])@X)
        return self.log_norm_constant()[:,None] -self.c*np.log(pdf)

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
        Q = np.sqrt(weights)[:,None,None]*X
        
        # iteration 0 (initialized Sigma=eye(p)):
        # Sigma = p*np.sum(Q@np.swapaxes(Q,-2,-1),axis=0) \
        #         /(q*np.sum(weights))

        j = 0
        while np.linalg.norm(Sigma_old-Sigma) > tol and (j < max_iter):
            Sigma_old = Sigma
            
            # this has been tested in the "Naive" version below
            XtLX = np.swapaxes(X,-2,-1)@np.linalg.inv(Sigma)@X
            XtLX_trace = np.sum(1/np.linalg.eigh(XtLX)[0],axis=1)
            
            Sigma = p*np.sum(Q@np.linalg.inv(XtLX)@np.swapaxes(Q,-2,-1),axis=0) \
                /(np.sum(weights*XtLX_trace))
            j +=1
        return Sigma


############# M-step #################
    def M_step(self,X,tol=1e-10):
        n,p,q = X.shape
        Beta = np.exp(self.density-self.logsum_density).T
        self.pi = np.sum(Beta,axis=0)/n

        for k in range(self.K):
            self.Sigma[k] = self.Sigma_MLE(self.Sigma[k],X,weights=Beta[:,k],tol=tol)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    K = np.array(2)
    
    p = np.array(3)
    MACG = MACG(K=K,p=3,q=2)
    data = np.loadtxt('data/synthetic/synth_data_4.csv',delimiter=',')
    data2 = np.zeros((1000,p,2))
    data2[:,:,0] = data[np.arange(2000,step=2),:]
    data2[:,:,1] = data[np.arange(2000,step=2)+1,:]
    # data = np.random.normal(loc=0,scale=0.1,size=(10000,100))
    # data = data[np.arange(2000,step=2),:]
    MACG.initialize(X=data2,init='uniform')
    # ACG.Lambda_MLE(X=data)

    # start = time.time()
    # ACG.log_norm_constant()
    # stop1 = time.time()-start
    # start = time.time()
    # ACG.log_norm_constant2()
    # stop2 = time.time()-start
    # print(str(stop1)+"_"+str(stop2))


    for iter in tqdm(range(1000)):
        # E-step
        MACG.log_likelihood(X=data2)
        # print(ACG.Lambda_chol)
        # M-step
        MACG.M_step(X=data2)
    stop=7


    
    # def Sigma_MLE_naive(self,X,weights = None,tol=1e-10,max_iter=10000):
    #     n,p,q = X.shape
    #     if weights is None:
    #         weights = np.ones(n)
    #     Sigma = np.eye(self.p)
    #     Sigma_old = Sigma + 10000
    #     # Q = np.sqrt(weights)[:,np.newaxis]*X
        
    #     j = 0
    #     while np.linalg.norm(Sigma_old-Sigma) > tol and (j < max_iter):
    #         Sigma_old = Sigma
    #         tmp = np.zeros((p,p))
    #         tmp2 = np.zeros((p,p))
    #         tmp3 = 0
    #         tmp4 = 0
    #         for i in range(n):
    #             tmp += p/(n*q)*X[i]@np.linalg.inv(X[i].T@np.linalg.inv(Sigma)@X[i])@X[i].T
    #             tmp2 += X[i]@np.linalg.inv(X[i].T@np.linalg.inv(Sigma)@X[i])@X[i].T
    #             # tmp3 += 1/(np.trace(X[i].T@np.linalg.inv(Sigma)@X[i]))
    #             tmp4 += np.sum(1/np.linalg.eigh(X[i].T@np.linalg.inv(Sigma)@X[i])[0])


    #         XtLX = np.swapaxes(X,-2,-1)@np.linalg.inv(Sigma)@X
    #         XtLX_trace = np.sum(1/np.linalg.eigh(XtLX)[0],axis=1)
    #         Sigma2 = np.sum(XtLX_trace)
            
    #         Sigma1 = np.sum(X@np.linalg.inv(XtLX)@np.swapaxes(X,-2,-1),axis=0)
            
    #         Sigma_mat = p*Sigma1/Sigma2
            
    #         # known true results
    #         Sigma_iter = p*tmp2/tmp4
    #         Sigma = p/np.trace(tmp)*tmp
    #         j +=1
    #     return Sigma