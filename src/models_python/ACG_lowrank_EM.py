import numpy as np
from scipy.special import loggamma
from src.models_python.diametrical_clustering import diametrical_clustering, diametrical_clustering_plusplus
from src.models_python.WatsonMixtureEM import Watson
from src.models_python.mixture_EM_loop import mixture_EM_loop
import matplotlib.pyplot as plt

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
        self.d = 1
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
        return {'M': self.M,'pi':self.pi}
    
    def initialize(self,X=None,init=None,tol=None):
        self.pi = np.repeat(1/self.K,repeats=self.K)
        if init is not None and init!='uniform' and init!='unif':
            if init == '++' or init == 'plusplus' or init == 'diametrical_clustering_plusplus':
                mu = diametrical_clustering_plusplus(X=X,K=self.K)
            elif init == 'dc' or init == 'diametrical_clustering':
                mu = diametrical_clustering(X=X,K=self.K,max_iter=100000,num_repl=5,init='++',tol=tol)
            elif init == 'WMM' or init == 'Watson' or init == 'W' or init == 'watson':
                W = Watson(K=self.K,p=self.p)
                params,_,_,_ = mixture_EM_loop(W,X,init='dc')
                mu = params['mu']
                self.pi = params['pi']
            
            self.M = np.random.uniform(size=(self.K,self.p,self.r))
            for k in range(self.K):
                self.M[k,:,0] = mu[:,k] #initialize only the first of the rank D columns this way, the rest uniform
        elif init =='unif' or init=='uniform' or init is None:
            self.M = np.random.uniform(size=(self.K,self.p,self.r))
        
################ E-step ###################
    
    def log_norm_constant(self,Lambda):
        logdetsign,logdet = np.linalg.slogdet(Lambda)
        return self.logSA - 0.5*logdetsign*logdet

    def log_pdf(self,X):
        Lambda = self.d*np.eye(self.r) + np.swapaxes(self.M,-2,-1)@self.M
        B = X[None,:,:]@self.M
        pdf = 1-np.sum(B@np.linalg.inv(Lambda)*B,axis=2) #check

        # minus log_det_L instead of + log_det_A_inv
        log_acg_pdf = self.log_norm_constant(Lambda)[:,None] - self.half_p * np.log(pdf)
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
            print("Too high dimensionality compared to number of observations. Lambda cannot be calculated")
            return
        weights = None
        if weights is None:
            weights = np.ones(n)
        M_old = np.ones((self.p,self.r))
        Q = np.sqrt(weights)[:,None]*X

        j = 0
        loss = []
        c = d = 1

        # not woodbury
        

        # Woodbury
        q = np.linalg.norm(M,'fro')**2
        b = 1/(c+q/p)
        M = np.sqrt(b)*M
        c = b*c
        D = np.eye(self.r)+1/c*M.T@M
        Lambda2 = c*np.eye(self.p)+M@M.T

        Lambda=Lambda2
        Lambda_old = 100000*Lambda.copy()
        
        while (j < max_iter) and np.linalg.norm(Lambda_old-Lambda) > tol:
            # plt.figure(),plt.imshow(M@M.T+np.eye(self.p)),plt.colorbar()
            Lambda_old = Lambda

            # Woodbury scaled
            D_inv = np.linalg.inv(np.eye(self.r)+1/c*M.T@M)
            XM = X@M
            XMLMtX = 1/c-1/c**2*np.sum(XM@D_inv*XM,axis=1) #denominator
            M = c**-1*p/np.sum(weights)*Q.T/XMLMtX@Q@M@D_inv
            q = np.linalg.norm(M,'fro')**2
            b = 1/(c+q/p)
            M = np.sqrt(b)*M
            c = b*c

            # only used for convergence check, could be avoided using M instead....
            Lambda= c*np.eye(self.p)+M@M.T
            
            j +=1
            if j>1:
                loss.append(np.linalg.norm(Lambda_old-Lambda))
        self.d = d
        return M
    

    

############# M-step #################
    def M_step(self,X,tol=1e-10):
        n,_ = X.shape
        Beta = np.exp(self.density-self.logsum_density).T
        self.pi = np.sum(Beta,axis=0)/n

        for k in range(self.K):
            self.M[k] = self.M_MLE_lowrank(self.M[k],X,weights=Beta[:,k],tol=tol)


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