import numpy as np
from src.DMM_EM.diametrical_clustering import diametrical_clustering, diametrical_clustering_plusplus

class DMMEMBaseModel():

    def __init__(self):
        super().__init__()

    def unpack_params(self,params):
        
        # mixture settings
        if 'pi' in params:
            self.pi = params['pi']
        else:
            self.pi = np.array([1/self.K]*self.K)

        # distribution-specific settings
        if self.distribution == 'Watson':
            self.mu = params['mu']
            self.kappa = params['kappa']
        elif self.distribution == 'ACG_lowrank' or self.distribution == 'MACG_lowrank':
            M_init = params['M']
                
            if M_init.ndim!=3 or M_init.shape[2]!=self.r: # add extra columns
                if M_init.ndim==2:
                    num_missing = self.r-1
                else:
                    num_missing = self.r-M_init.shape[2]

                M_extra = np.random.uniform((self.K,M_init.shape[1],num_missing))
                self.M = np.concatenate([M_init,M_extra],axis=2)
            else:
                self.M = M_init
        elif self.distribution == 'ACG_fullrank' or self.distribution == 'MACG_fullrank':
            self.Lambda = params['Lambda']
        else:
            raise ValueError('Invalid distribution')

    def initialize(self,X,init_method):
        self.pi = np.array([1/self.K]*self.K)
        if X.ndim==3:
            X = X[:,:,0]
        if init_method=='uniform' or init_method=='unif':
            mu = np.random.uniform(size=(self.p,self.K))
            mu = mu / np.linalg.norm(mu,axis=0)
        elif init_method == '++' or init_method == 'plusplus' or init_method == 'diametrical_clustering_plusplus' or init_method=='dc++':
            mu = diametrical_clustering_plusplus(X=X,K=self.K)
        elif init_method == 'dc' or init_method == 'diametrical_clustering':
            mu = diametrical_clustering(X=X,K=self.K,max_iter=100000,num_repl=5,init='++')
        else:
            raise ValueError('Invalid init_method')

        if self.distribution == 'Watson':
            self.mu = mu
            self.kappa = np.ones(self.K)
        elif self.distribution == 'ACG_lowrank' or self.distribution=='MACG_lowrank':
            self.M = np.random.uniform(size=(self.K,self.p,self.r))
            if init_method is not None and init_method !='unif' and init_method!='uniform':
                for k in range(self.K):
                    self.M[k,:,0] = mu[:,k] #initialize only the first of the rank D columns this way, the rest uniform
        elif self.distribution == 'ACG_fullrank' or self.distribution=='MACG_fullrank':
            self.Lambda = np.zeros((self.K,self.p,self.p))    
            for k in range(self.K):
                self.Lambda[k] = 10e6*np.outer(mu[:,k],mu[:,k])+np.eye(self.p)
                self.Lambda[k] = self.Lambda[k]/np.trace(self.Lambda[k])

    def logdet(self,B):
        logdetsign,logdet = np.linalg.slogdet(B)
        return logdetsign*logdet

    # here the log_pdf should be formatted as KxN
    def MM_log_likelihood(self,log_pdf):
        if log_pdf.ndim!=2:
            raise ValueError("log_pdf should be KxN, where N is concatenated across all subjects/batches")
        log_density = log_pdf+np.log(self.pi)[:,None] #each pdf gets "multiplied" with the weight
        logsum_density = np.logaddexp.reduce(log_density) #sum over the K components
        log_likelihood = np.sum(logsum_density) #sum over the N samples
        return log_likelihood,log_density,logsum_density

    def log_likelihood(self, X):
        log_pdf = self.log_pdf(X)
        if self.K==1:
            return np.sum(log_pdf)
        else:
            log_likelihood, self.log_density, self.logsum_density = self.MM_log_likelihood(log_pdf)
            return log_likelihood
        
    def test_log_likelihood(self,X):
        return self.log_likelihood(X)

    def update_pi(self,Beta):
        self.pi = np.sum(Beta,axis=1)/Beta.shape[1]

    def posterior_MM(self,log_pdf):
        log_density = log_pdf+np.log(self.pi)[:,None]
        logsum_density = np.logaddexp.reduce(log_density)
        return np.exp(log_density-logsum_density)
    
    def posterior(self,X):
        log_pdf = self.log_pdf(X)
        return self.posterior_MM(log_pdf)

    def get_params(self):
        if self.distribution == 'Watson':
            return {'mu':self.mu,'kappa':self.kappa,'pi':self.pi}
        elif self.distribution == 'ACG_lowrank' or self.distribution == 'MACG_lowrank':
            return {'M':self.M,'pi':self.pi}
        elif self.distribution == 'ACG_fullrank' or self.distribution == 'MACG_fullrank':
            return {'Lambda':self.Lambda,'pi':self.pi}

    def sample(self, num_samples):
        raise NotImplementedError("Subclasses must implement sample() method")
