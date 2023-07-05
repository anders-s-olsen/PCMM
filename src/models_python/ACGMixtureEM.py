import numpy as np
from scipy.special import loggamma
from src.models_python.diametrical_clustering import diametrical_clustering, diametrical_clustering_plusplus
from src.models_python.WatsonMixtureEM import Watson
from src.models_python.mixture_EM_loop import mixture_EM_loop

class ACG():
    """
    Mixture model class
    """
    def __init__(self, K: int, p: int,params=None):
        super().__init__()
        self.K = K
        self.p = p
        self.c = self.p/2
        self.logSA = loggamma(self.c) - np.log(2) -self.c* np.log(np.pi)

        if params is not None: # for evaluating likelihood with already-learned parameters
            self.Lambda = np.array(params['Lambda'])
            self.pi = np.array(params['pi'])

    def get_params(self):
        return {'Lambda': self.Lambda,'pi':self.pi}
    
    def initialize(self,X=None,init=None,tol=None):
        self.pi = np.repeat(1/self.K,repeats=self.K)
        if init is None or init=='uniform' or init=='unif':
            mu = np.random.uniform(size=(self.p,self.K))
            mu = mu/np.linalg.norm(mu,axis=0)
        elif init == '++' or init == 'plusplus' or init == 'diametrical_clustering_plusplus':
            mu = diametrical_clustering_plusplus(X=X,K=self.K)
        elif init == 'dc' or init == 'diametrical_clustering':
            mu = diametrical_clustering(X=X,K=self.K,max_iter=100000,num_repl=5,init='++',tol=tol)
        elif init == 'WMM' or init == 'Watson' or init == 'W' or init == 'watson':
            W = Watson(K=self.K,p=self.p)
            params,_,_,_ = mixture_EM_loop(W,X,init='dc')
            mu = params['mu']
            self.pi = params['pi']
        self.Lambda = np.zeros((self.K,self.p,self.p))    
        for k in range(self.K):
            self.Lambda[k] = np.outer(mu[:,k],mu[:,k])+np.eye(self.p)
        
################ E-step ###################
    
    def log_norm_constant(self):
        norm_const = self.logSA - 0.5*np.log(np.linalg.det(self.Lambda))
        return norm_const

    def log_pdf(self,X):
        pdf = np.zeros((self.K,X.shape[0]))
        for k in range(self.K):
            pdf[k] = np.sum(X@np.linalg.inv(self.Lambda[k])*X,axis=1)
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

    def Lambda_MLE(self,Lambda,X,weights = None,tol=1e-10,max_iter=10000):
        n,p = X.shape
        if n<p*(p-1):
            print("Too high dimensionality compared to number of observations. Lambda cannot be calculated")
            return
        if weights is None:
            weights = np.ones(n)
        Lambda_old = np.eye(self.p)
        Q = np.sqrt(weights)[:,None]*X

        # iteration 0 (Lambda initialized as eye(p)):
        # Lambda = p*Q.T@Q/np.sum(weights)
        # update: Lambda now initialized as old Lambda
        
        j = 0
        while np.linalg.norm(Lambda_old-Lambda) > tol and (j < max_iter):
            Lambda_old = Lambda
            
            # The following has been tested against a for-loop implementing Tyler1987 (3)
            # and also an implementation of Tyler1987 (2) where (2) has been evaluated and
            # Lambda=p/trace(Lambda)*Lambda for each iteration. All give the same result, 
            # at least before implementing weights

            XtLX = np.sum(X@np.linalg.inv(Lambda)*X,axis=1) #x_i^T*L^(-1)*x_i for batch
            Lambda = p*(Q/XtLX[:,np.newaxis]).T@Q/np.sum(weights/XtLX)
            j +=1
        return Lambda
    

    

############# M-step #################
    def M_step(self,X,tol=1e-10):
        n,_ = X.shape
        Beta = np.exp(self.density-self.logsum_density).T
        self.pi = np.sum(Beta,axis=0)/n

        for k in range(self.K):
            self.Lambda[k] = self.Lambda_MLE(self.Lambda[k],X,weights=Beta[:,k],tol=tol)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    K = np.array(2)
    
    p = np.array(3)
    ACG = ACG(K=K,p=p)
    data = np.loadtxt('data/synthetic/synth_data_ACG.csv',delimiter=',')
    # data = np.random.normal(loc=0,scale=0.1,size=(10000,100))
    # data = data[np.arange(1000,step=2),:]
    ACG.initialize(X=data,init='uniform')
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
        ACG.log_likelihood(X=data)
        # print(ACG.Lambda_chol)
        # M-step
        ACG.M_step(X=data)
    stop=7

    def Lambda_MLE_naive(self,X,weights = None,tol=1e-10,max_iter=10000):
        n,p = X.shape
        if weights is None:
            weights = np.ones(n)
        Lambda = np.eye(self.p)
        Lambda_old = Lambda + 10000
        # Q = np.sqrt(weights)[:,np.newaxis]*X
        
        j = 0
        while np.linalg.norm(Lambda_old-Lambda) > tol and (j < max_iter):
            Lambda_old = Lambda
            tmp = np.zeros((p,p))
            # tmp2 = np.zeros((p,p))
            # tmp3 = 0
            Lambda_inv = np.linalg.inv(Lambda)
            for i in range(n):
                tmp += p/n*np.outer(X[i],X[i])/(X[i]@Lambda_inv@X[i])
                # tmp2 += np.outer(X[i],X[i])/(X[i]@np.linalg.inv(Lambda)@X[i])
                # tmp3 += 1/(X[i]@np.linalg.inv(Lambda)@X[i])
            # Lambda_iter = p*tmp2/tmp3
            Lambda = p/np.trace(tmp)*tmp
            j +=1
            print(j)
        return Lambda
    
    # def Lambda_MLE_chol(self,X,weights=None,tol=1e-10,max_iter=10000):
    #     n,p = X.shape
    #     if weights is None:
    #         weights = np.ones(n)
    #     Lambda = np.eye(self.p)
    #     Lambda_old = Lambda + np.eye(self.p)*10000
    #     Q = np.sqrt(weights)[:,np.newaxis]*X
        
    #     j = 0
    #     while np.linalg.norm(Lambda_old-Lambda) > tol and (j < max_iter):
    #         Lambda_old = Lambda
            
    #         B = X @ Lambda
    #         XLXt = np.sum(B * B, axis=1)
    #         Lambda = np.linalg.cholesky(p*(Q/XLXt[:,np.newaxis]).T@Q/np.sum(weights/XLXt))
    #         j +=1
    #     return Lambda


    
    
    # def log_norm_constant_chol(self):
    #     # Be sure to check with task code
    #     logdet = 2*np.log(np.linalg.det(self.Lambda_chol))
    #     logdet = np.zeros(self.K)
    #     for k in range(self.K):
    #         logdet[k] = 2*np.sum(np.log(np.abs(np.diag(self.Lambda_chol[k]))))
    #     return self.logSA - 0.5*logdet