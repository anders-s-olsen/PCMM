import numpy as np
from scipy.special import loggamma,hyp1f1
import scipy
import time
from diametrical_clustering import diametrical_clustering, diametrical_clustering_plusplus
from WatsonMixtureEM import Watson

class Watson():
    """
    Mixture model class
    """
    def __init__(self, K: int, p: int):
        super().__init__()
        self.K = K
        self.p = p
        self.c = self.p/2
        self.a = 0.5
        self.log_a_div_c = np.log(self.a/self.c)
        self.logSA = loggamma(self.c) - np.log(2) -self.c* np.log(np.pi)
        self.loglik = []

    def get_parameters(self):
        return {'mu': self.mu,'kappa':self.kappa,'pi':self.pi}
    
    def initialize(self,X=None,init=None):
        if init is None or init=='uniform' or init=='unif':
            self.mu = np.random.uniform(size=(self.p,self.K))
            self.mu = self.mu/np.linalg.norm(self.mu,axis=0)
        elif init == "test":
            self.mu = np.vstack((np.array([1,1,1]),np.array([1,1,-1]))).T
            self.mu = self.mu/np.linalg.norm(self.mu,axis=0)
        elif init == '++' or init == 'plusplus' or init == 'diametrical_clustering_plusplus':
            self.mu = diametrical_clustering_plusplus(X=X,K=self.K)
        elif init == 'dc' or init == 'diametrical_clustering':
            self.mu,_,_ = diametrical_clustering(X=X,K=self.K,max_iter=100000,num_repl=5,init='++')
            
        self.pi = np.repeat(1/self.K,repeats=self.K)
        self.Lambda = self.mu.T@self.mu
    

################ E-step ###################
    
    
    def log_norm_constant(self):
        # Be sure to check with task code
        L = np.linalg.cholesky(self.Lambda)
        logdet = 2*np.trace(np.log(L))
        return self.logSA - logdet
    
    def log_pdf(self,X):
        pdf = np.zeros((self.K,X.size[0]))
        for k in range(self.K):
            pdf[k] = np.diag(X@self.Lambda[k]@X.T)
        return self.log_norm_constant() -self.c*np.log(pdf)

    def log_density(self,X):
        return self.log_pdf(X)+np.log(self.pi)[:,np.newaxis]
    
    def log_likelihood(self,X):
        self.density = self.log_density(X)
        self.logsum_density = np.logaddexp.reduce(self.density)
        # log(sum(exp(density-max(density))))+max(density);
        loglik = np.sum(self.logsum_density)
        self.loglik.append(loglik)
        return loglik

    def Lambda_update(self,X,tol=1e-10):
        self.Lambda[k] = np.eye()
        self.p*np.sum()


############# M-step #################
    def M_step(self,X):
        n,_ = X.shape
        Beta = np.exp(self.density-self.logsum_density).T
        self.pi = np.sum(Beta,axis=0)/n

        for k in range(self.K):
            Q = np.sqrt(Beta[:,k])[:,np.newaxis]*X

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

            # def f(kappa):
            #     return -(kappa * rk - hyp1f1(self.a, self.c, kappa))
            def f(kappa):
                return ((self.a/self.c)*(np.exp(self.kummer_log(self.a+1,self.c+1,kappa)-self.kummer_log(self.a,self.c,kappa)))-rk)**2

            if rk>self.a/self.c:
                self.kappa[k] = scipy.optimize.minimize(f, x0=np.mean([LB,B]), bounds=[(LB, B)])['x']
            elif rk<self.a/self.c:
                self.kappa[k] = scipy.optimize.minimize(f, x0=np.mean([B,UB]), bounds=[(B, UB)])['x']
            elif rk==self.a/self.c:
                self.kappa[k] = 0
            else:
                print("kappa could not be optimized")
                return


if __name__=='__main__':
    import matplotlib.pyplot as plt
    K = np.array(2)
    
    p = np.array(3)
    W = Watson(K=K,p=p)
    data = np.loadtxt('data/synthetic/synth_data_2.csv',delimiter=',')
    data = data[np.arange(2000,step=2),:]
    W.initialize(X=data,init='test')

    # data1 = np.random.normal(loc=np.array((1,1,1)),scale=0.02,size=(1000,p))
    # data2 = np.random.normal(loc=np.array((1,-1,-1)),scale=0.02,size=(1000,p))
    # data = np.vstack((data1,data2))
    # data = data/np.linalg.norm(data,axis=1)[:,np.newaxis]

    for iter in range(1000):
        # E-step
        W.log_likelihood(X=data)
        # M-step
        W.M_step(X=data)
    stop=7