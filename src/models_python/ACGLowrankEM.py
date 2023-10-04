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
        self.c = np.ones(self.K)
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
    
    # def log_norm_constant(self,Z):
    #     logdetsign,logdet = np.linalg.slogdet(Z)
    #     return self.logSA - 0.5*logdetsign*logdet
    
    def log_norm_constant_matrixdeterminantlemma(self,D):
        logdetsign,logdet = np.linalg.slogdet(D)
        return self.logSA - 0.5*(self.p*np.log(self.c)+logdetsign*logdet)

    def log_pdf(self,X):
        D = np.eye(self.r)+1/self.c[:,None,None]*np.swapaxes(self.M,-2,-1)@self.M
        XM = X[None,:,:]@self.M
        v = self.c[:,None]**(-1)-self.c[:,None]**(-2)*np.sum(XM@np.linalg.inv(D)*XM,axis=2) #check

        # for checking
        # Z = np.array([self.c[k]*np.eye(self.p) for k in range(self.K)]) + self.M@np.swapaxes(self.M,-2,-1)
        # v2 = np.sum(X@np.linalg.inv(Z)*X,axis=2)
        # log_acg_pdf2 = self.log_norm_constant(Z)[:,None] - self.half_p * np.log(v2)

        log_acg_pdf = self.log_norm_constant_matrixdeterminantlemma(D)[:,None] - self.half_p * np.log(v)
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
    
    def M_MLE_lowrank(self,M,X,weights = None,c=10e-6,tol=1e-10,max_iter=10000):
        n,p = X.shape
        if n<p*(p-1):
            print("Too high dimensionality compared to number of observations. Lambda cannot be calculated")
            return
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
            self.M[k],self.c[k] = self.M_MLE_lowrank(self.M[k],X,weights=Beta[:,k],c=self.c[k],tol=tol)
            


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