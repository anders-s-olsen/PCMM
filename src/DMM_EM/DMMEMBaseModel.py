import numpy as np
from src.DMM_EM.diametrical_clustering import diametrical_clustering, diametrical_clustering_plusplus, grassmannian_clustering_gruber2006

class DMMEMBaseModel():

    def __init__(self):
        super().__init__()

    def unpack_params(self,params):

        # distribution-specific settings
        if self.distribution == 'Watson':
            self.mu = params['mu']
            self.kappa = params['kappa']
        elif self.distribution == 'ACG_lowrank' or self.distribution == 'MACG_lowrank':
            M_init = params['M']
            M_init = M_init/np.linalg.norm(M_init,axis=1)[:,None,:]
                
            if M_init.ndim!=3 or M_init.shape[2]!=self.r: # add extra columns
                if M_init.ndim==2:
                    num_missing = self.r-1
                else:
                    num_missing = self.r-M_init.shape[2]

                M_extra = np.random.uniform(size=(self.K,M_init.shape[1],num_missing))
                M_extra = M_extra/np.linalg.norm(M_extra,axis=1)[:,None,:]/1000 #unit norm
                self.M = np.concatenate([M_init,M_extra],axis=2)
            else:
                self.M = M_init
        elif self.distribution == 'ACG_fullrank' or self.distribution == 'MACG_fullrank':
            self.Lambda = params['Lambda']
        else:
            raise ValueError('Invalid distribution')
        
        # mixture settings
        if 'pi' in params:
            self.pi = params['pi']
        else:
            self.pi = np.array([1/self.K]*self.K)

    def initialize(self,X,init_method):
        assert init_method in ['uniform','unif',
                               '++','plusplus','diametrical_clustering_plusplus','dc++',
                               '++_seg','plusplus_seg','diametrical_clustering_plusplus_seg','dc++_seg',
                               'dc','diametrical_clustering',
                               'dc_seg','diametrical_clustering_seg',
                               'Grassmann','Grassmann_clustering',
                               'Grassmann_seg','Grassmann_clustering_seg']
        assert self.distribution in ['Watson','ACG_lowrank','ACG_fullrank','MACG_lowrank','MACG_fullrank']

        # strategy: 
        # 1) if uniform, initialize with uniform random values and return
        # 2) if not, compute mu or C using dc++, dc, gr or gr++
        # 3) if not seg use those values to initialize the model
        # 4) if seg, use the partition and run the model a single time using the partition

        # initialize pi as uniform, will be overwritten by 'seg' methods
        self.pi = np.array([1/self.K]*self.K)

        # for watson, always initialize kappa as ones, will be overwritten by 'seg' methods
        if self.distribution == 'Watson':
            self.kappa = np.ones(self.K)

        if init_method in ['uniform','unif']:
            if self.distribution == 'Watson':
                mu = np.random.uniform(size=(self.p,self.K))
                self.mu = mu / np.linalg.norm(mu,axis=0)
            elif self.distribution in ['ACG_lowrank', 'MACG_lowrank']:
                self.M = np.random.uniform(size=(self.K,self.p,self.r))
            elif self.distribution in ['ACG_fullrank', 'MACG_fullrank']:
                M = np.random.uniform(size=(self.K,self.p,self.p))
                self.Lambda = np.zeros((self.K,self.p,self.p))
                for k in range(self.K):
                    self.Lambda[k] = M[k]@M[k].T+np.eye(self.p)
                    self.Lambda[k] = self.Lambda[k]/np.trace(self.Lambda[k])
            return
        elif init_method in ['++','plusplus','diametrical_clustering_plusplus','dc++','++_seg','plusplus_seg','diametrical_clustering_plusplus_seg','dc++_seg','dc','diametrical_clustering','dc_seg','diametrical_clustering_seg']:
            if X.ndim==3:
                X2 = X[:,:,0]
            else:
                X2 = X.copy()   
            
            # if '++' or 'plusplus' in init_method
            if init_method in ['++','plusplus','diametrical_clustering_plusplus','dc++','++_seg','plusplus_seg','diametrical_clustering_plusplus_seg','dc++_seg']:
                mu = diametrical_clustering_plusplus(X=X2,K=self.K)
            elif init_method in ['dc','diametrical_clustering','dc_seg','diametrical_clustering_seg']:
                mu = diametrical_clustering(X=X2,K=self.K,max_iter=100000,num_repl=1,init='++')
            
            if self.distribution == 'Watson':
                self.mu = mu
            elif self.distribution in ['ACG_lowrank','MACG_lowrank']:
                self.M = np.random.uniform(size=(self.K,self.p,self.r))
                self.M = self.M/np.linalg.norm(self.M,axis=1)[:,None,:] #unit norm
                for k in range(self.K):
                    self.M[k,:,0] = 1000*mu[:,k] #upweight compared to the random columns
            elif self.distribution in ['ACG_fullrank', 'MACG_fullrank']:
                self.Lambda = np.zeros((self.K,self.p,self.p))
                for k in range(self.K):
                    self.Lambda[k] = np.outer(mu[:,k],mu[:,k])+np.eye(self.p)
                    self.Lambda[k] = self.Lambda[k]/np.trace(self.Lambda[k])

            # if segmentation methods, use mu to segment the data and later estimate single-component models
            if init_method in ['++_seg','plusplus_seg','diametrical_clustering_plusplus_seg','dc++_seg','dc_seg','diametrical_clustering_seg']:
                dis = (X2@mu)**2           
                X_part = np.argmax(dis,axis=1) 
            
        elif init_method in ['Grassmann','Grassmann_clustering','Grassmann_seg','Grassmann_clustering_seg']:
            if X.ndim!=3:
                raise ValueError('Grassmann methods are only implemented for 3D data')
            
            C = grassmannian_clustering_gruber2006(X=X,K=self.K,max_iter=100000,num_repl=1)
            
            if self.distribution == 'MACG_lowrank':
                self.M = np.random.uniform(size=(self.K,self.p,self.r))
                self.M = self.M/np.linalg.norm(self.M,axis=1)[:,None,:] #unit norm
                for k in range(self.K):
                    self.M[k,:,0] = 1000*C[k,:,0] #upweight compared to the random columns
            elif self.distribution == 'MACG_fullrank':
                self.Lambda = np.zeros((self.K,self.p,self.p))
                for k in range(self.K):
                    self.Lambda[k] = C[k]@C[k].T+np.eye(self.p)
                    self.Lambda[k] = self.Lambda[k]/np.trace(self.Lambda[k])

            if init_method in ['Grassmann_seg','Grassmann_clustering_seg']:
                XXt = X@np.swapaxes(X,-2,-1)
                CCt = C@np.swapaxes(C,-2,-1)
                dis = 1/np.sqrt(2)*np.linalg.norm(XXt[:,None]-CCt[None],axis=(-2,-1))
                X_part = np.argmin(dis,axis=1)
        else:
            raise ValueError('Invalid init_method')
            
        if init_method in ['++_seg','plusplus_seg','diametrical_clustering_plusplus_seg','dc++_seg','dc_seg','diametrical_clustering_seg','Grassmann_seg','Grassmann_clustering_seg']:
            print('Running single component models as initialization')
            self.pi = np.bincount(X_part)/X_part.shape[0]
            # run a single-component model on each segment
            if self.distribution=='Watson':
                mu = np.zeros((self.p,self.K))
                kappa = np.zeros(self.K)
                for k in range(self.K):
                    mu[:,k],kappa[k] = self.M_step_single_component(X=X[X_part==k],Beta=np.ones(np.sum(X_part==k)),mu=self.mu[:,k],kappa=1)
                self.mu = mu
                self.kappa = kappa
            elif self.distribution in ['ACG_lowrank','MACG_lowrank']:
                M = np.zeros((self.K,self.p,self.r))
                for k in range(self.K):
                    M[k] = self.M_step_single_component(X=X[X_part==k],Beta=np.ones(np.sum(X_part==k)),M=self.M[k],Lambda=None)
                self.M = M
            elif self.distribution in ['ACG_fullrank','MACG_fullrank']:
                Lambda = np.zeros((self.K,self.p,self.p))
                for k in range(self.K):
                    Lambda[k] = self.M_step_single_component(X=X[X_part==k],Beta=np.ones(np.sum(X_part==k)),M=None, Lambda=self.Lambda[k])
                self.Lambda=Lambda

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
        
    def set_params(self,params):
        self.unpack_params(params)

    def sample(self, num_samples):
        raise NotImplementedError("Subclasses must implement sample() method")
