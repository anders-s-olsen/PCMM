import numpy as np
from src.DMM_EM.riemannian_clustering import diametrical_clustering, plusplus_initialization, grassmann_clustering, weighted_grassmann_clustering

class DMMEMBaseModel():

    def __init__(self):
        super().__init__()
    
    def unpack_params(self,params):

        # distribution-specific settings
        if self.distribution == 'Watson':
            self.mu = params['mu']
            self.kappa = params['kappa']
        elif self.distribution in ['ACG_lowrank','MACG_lowrank','SingularWishart_lowrank']:
            self.M = params['M']
            if self.M.ndim==2 and self.K==1:
                self.M = self.M[None,:,:]
        elif self.distribution in ['ACG_fullrank','MACG_fullrank','SingularWishart_fullrank']:
            self.Lambda = params['Lambda']
        else:
            raise ValueError('Invalid distribution')
        
        if self.distribution == 'SingularWishart_lowrank':
            if 'gamma' in params:
                self.gamma = params['gamma']
            else:
                self.gamma = np.ones(self.K)

        # mixture settings
        if 'pi' in params:
            self.pi = params['pi']
        else:
            self.pi = np.array([1/self.K]*self.K)

    def init_M_svd(self,V,r):
        U,S,_ = np.linalg.svd(V,full_matrices=False)
        # if V.shape[0]>=self.p:
        epsilon = np.sum(S[r:])/(self.p-r)
        # else:
        #     epsilon = np.sum(S[r:])/(V.shape[0]-r)
        M = U[:,:r]@np.diag(np.sqrt((S[:r]-epsilon)/epsilon))

        # if V.shape[0]>=self.p:
        #     sigma = np.sum(S[r:]**2)/(self.p-r)
        # else:
        #     sigma = np.sum(S[r:]**2)/(V.shape[0]-r)
        # M = U[:,:r]@np.diag((S[:r]-sigma)/sigma)
        
        return M
    
    def init_M_svd_given_M_init(self,X,M_init,beta=None):
        if beta is None:
            beta = np.ones((self.K,X.shape[0]))/self.K
        elif beta.ndim==1:
            raise ValueError('beta should be KxN')

        # M_init should be Kxpxr_1, where r_1<r.
        if M_init.ndim==2:
            M_init = M_init[:,:,None]
        num_missing = self.r-M_init.shape[2]

        if self.r == 1:
            raise ValueError('r=1 not implemented')
        
        # #orthgonalize M_init
        # M_init_orth,_ = np.linalg.qr(M_init)

        # initialize remainder using the svd of the residual of X after subtracting the projection on M_init
        M = np.zeros((self.K,self.p,self.r))
        for k in range(self.K):
            if X.ndim==2:
                V = (beta[k][:,None]*X).T
            else:
                V = np.reshape(np.swapaxes(beta[k][:,None,None]*X,0,1),(self.p,self.q*X.shape[0]))
                
            # projection matrix of M
            # V_residual = V-M_init_orth[k]@np.swapaxes(M_init_orth[k],-2,-1)@V
            # V_residual = V-M_init[k]@np.swapaxes(M_init[k],-2,-1)@V
            M_proj = M_init[k]@np.linalg.inv(M_init[k].T@M_init[k])@M_init[k].T
            V_residual = V - M_proj@V
            M_extra = self.init_M_svd(V_residual,r=num_missing)
            M[k] = np.concatenate([M_init[k],M_extra],axis=-1)
        return M

    def initialize(self,X,init_method,L=None):
        assert init_method in ['uniform','unif',
                               'diametrical_clustering_plusplus','dc++','diametrical_clustering_plusplus_seg','dc++_seg',
                               'grassmann_clustering_plusplus','gc++','grassmann_clustering_plusplus_seg','gc++_seg',
                               'weighted_grassmann_clustering_plusplus','wgc++','weighted_grassmann_clustering_plusplus_seg','wgc++_seg',
                               'dc','diametrical_clustering','dc_seg','diametrical_clustering_seg',
                               'gc','grassmann_clustering','gc_seg','grassmann_clustering_seg',
                               'wgc','weighted_grassmann_clustering','wgc_seg','weighted_grassmann_clustering_seg']
        assert self.distribution in ['Watson','ACG_lowrank','ACG_fullrank','MACG_lowrank','MACG_fullrank','SingularWishart_lowrank','SingularWishart_fullrank']

        # for watson, always initialize kappa as ones, will be overwritten by 'seg' methods
        if self.distribution == 'Watson':
            self.kappa = np.ones(self.K)
        elif self.distribution == 'SingularWishart_lowrank':
            self.gamma = np.ones(self.K)

        if init_method in ['uniform','unif']:
            print('Initializing parameters using the uniform distribution')
            self.pi = np.array([1/self.K]*self.K)
            if self.distribution == 'Watson':
                mu = np.random.uniform(size=(self.p,self.K))
                self.mu = mu / np.linalg.norm(mu,axis=0)
            elif self.distribution in ['ACG_lowrank', 'MACG_lowrank','SingularWishart_lowrank']:
                self.M = np.random.uniform(size=(self.K,self.p,self.r))
            elif self.distribution in ['ACG_fullrank', 'MACG_fullrank','SingularWishart_fullrank']:
                M = np.random.uniform(size=(self.K,self.p,self.p))
                self.Lambda = np.zeros((self.K,self.p,self.p))
                for k in range(self.K):
                    self.Lambda[k] = M[k]@M[k].T+np.eye(self.p)
                    self.Lambda[k] = self.p*self.Lambda[k]/np.trace(self.Lambda[k])
            return
        elif init_method in ['diametrical_clustering_plusplus','dc++','dc++_seg','diametrical_clustering_plusplus_seg','dc','diametrical_clustering','dc_seg','diametrical_clustering_seg']:
            if X.ndim==3:
                X2 = X[:,:,0]
            else:
                X2 = X.copy()   
            
            if init_method in ['diametrical_clustering_plusplus','dc++','diametrical_clustering_plusplus_seg','dc++_seg']:
                print('Running diametrical clustering ++ initialization')
                mu,X_part,_ = plusplus_initialization(X=X2,K=self.K)
            elif init_method in ['dc','diametrical_clustering','dc_seg','diametrical_clustering_seg']:
                print('Running diametrical clustering initialization')
                mu,X_part,_ = diametrical_clustering(X=X2,K=self.K,max_iter=100000,num_repl=1,init='++')
            
            if self.distribution == 'Watson':
                print('Initializing mu based on the clustering centroid')
                self.mu = mu
            elif self.distribution == 'ACG_lowrank':
                print('Initializing M based on a lowrank-svd of the input data partitioned acc to the clustering')
                self.M = np.zeros((self.K,self.p,self.r))
                for k in range(self.K):
                    self.M[k] = self.init_M_svd(X[X_part==k].T,self.r)
            elif self.distribution == 'MACG_lowrank':
                print('Initializing M based on a lowrank-svd of the input data partitioned acc to the clustering')
                self.M = np.zeros((self.K,self.p,self.r))
                for k in range(self.K):
                    self.M[k] = self.init_M_svd(np.reshape(np.swapaxes(X[X_part==k],0,1),(self.p,np.sum(X_part==k)*self.q)),self.r)
            elif self.distribution == 'SingularWishart_lowrank':
                print('Initializing M based on a lowrank-svd of the input data partitioned acc to the clustering')
                self.M = np.zeros((self.K,self.p,self.r))
                for k in range(self.K):
                    self.M[k] = self.init_M_svd(np.reshape(np.swapaxes(X[X_part==k]*np.sqrt(L[X_part==k][:,None,:]),0,1),(self.p,np.sum(X_part==k)*self.q)),self.r)
            elif self.distribution in ['ACG_fullrank', 'MACG_fullrank','SingularWishart_fullrank']:
                print('Initializing Lambda based on the clustering centroids')
                self.Lambda = np.zeros((self.K,self.p,self.p))
                for k in range(self.K):
                    self.Lambda[k] = np.outer(mu[:,k],mu[:,k])+np.eye(self.p)
                    self.Lambda[k] = self.p*self.Lambda[k]/np.trace(self.Lambda[k])
            
        elif init_method in ['grassmann_clustering','grassmann_clustering_seg','gc','gc_seg','grassmann_clustering_plusplus','gc++','grassmann_clustering_plusplus_seg','gc++_seg']:
            if X.ndim!=3:
                raise ValueError('Grassmann methods are only implemented for 3D data')
            
            if init_method in ['grassmann_clustering_plusplus','gc++','grassmann_clustering_plusplus_seg','gc++_seg']:
                print('Running grassmann clustering ++ initialization')
                C,X_part,_ = plusplus_initialization(X=X,K=self.K,dist='grassmann')
            elif init_method in ['gc','grassmann_clustering','gc_seg','grassmann_clustering_seg']:
                print('Running grassmann clustering initialization')
                C,X_part,_ = grassmann_clustering(X=X,K=self.K,max_iter=100000,num_repl=1)
            
            if self.distribution == 'MACG_lowrank':
                print('Initializing M based on a lowrank-svd of the input data partitioned acc to the clustering')
                self.M = np.zeros((self.K,self.p,self.r))
                for k in range(self.K):
                    self.M[k] = self.init_M_svd(np.reshape(np.swapaxes(X[X_part==k],0,1),(self.p,np.sum(X_part==k)*self.q)),self.r)
            elif self.distribution == 'SingularWishart_lowrank':
                print('Initializing M based on a lowrank-svd of the input data partitioned acc to the clustering')
                self.M = np.zeros((self.K,self.p,self.r))
                for k in range(self.K):
                    self.M[k] = self.init_M_svd(np.reshape(np.swapaxes(X[X_part==k]*np.sqrt(L[X_part==k][:,None,:]),0,1),(self.p,np.sum(X_part==k)*self.q)),self.r)
            elif self.distribution in ['MACG_fullrank','SingularWishart_fullrank']:
                print('Initializing Lambda based on the clustering centroids')
                self.Lambda = np.zeros((self.K,self.p,self.p))
                for k in range(self.K):
                    self.Lambda[k] = C[k]@C[k].T+np.eye(self.p)
                    self.Lambda[k] = self.p*self.Lambda[k]/np.trace(self.Lambda[k])

        elif init_method in ['weighted_grassmann_clustering','weighted_grassmann_clustering_seg','wgc','wgc_seg','weighted_grassmann_clustering_plusplus','wgc++','weighted_grassmann_clustering_plusplus_seg','wgc++_seg']:
            if X.ndim!=3:
                raise ValueError('Grassmann methods are only implemented for 3D data')
            if L is None:
                raise ValueError('L should be provided for weighted initialization')
            
            if init_method in ['weighted_grassmann_clustering_plusplus','wgc++','weighted_grassmann_clustering_plusplus_seg','wgc++_seg']:
                print('Running weighted grassmann clustering ++ initialization')
                C,C_weights,X_part,_ = plusplus_initialization(X=X,K=self.K,dist='weighted_grassmann',X_weights=L)
            elif init_method in ['wgc','weighted_grassmann_clustering','wgc_seg','weighted_grassmann_clustering_seg']:
                print('Running weighted grassmann clustering initialization')
                C,C_weights,X_part,_ = weighted_grassmann_clustering(X=X,X_weights=L,K=self.K,max_iter=100000,num_repl=1)
            
            if self.distribution == 'SingularWishart_lowrank':
                print('Initializing M based on a lowrank-svd of the input data partitioned acc to the clustering')
                self.M = np.zeros((self.K,self.p,self.r))
                for k in range(self.K):
                    self.M[k] = self.init_M_svd(np.reshape(np.swapaxes(X[X_part==k]*np.sqrt(L[X_part==k][:,None,:]),0,1),(self.p,np.sum(X_part==k)*self.q)),self.r)
                    # self.gamma[k] = 
            elif self.distribution == 'SingularWishart_fullrank':
                print('Initializing Lambda based on the clustering centroids')
                self.Lambda = np.zeros((self.K,self.p,self.p))
                for k in range(self.K):
                    self.Lambda[k] = C[k]@np.diag(C_weights[k])@C[k].T+np.eye(self.p)
        else:
            raise ValueError('Invalid init_method')
        
        self.pi = np.bincount(X_part)/X_part.shape[0]
            
        if '_seg' in init_method:
            print('Estimating single component models as initialization')
            # run a single-component model on each segment
            if self.distribution=='Watson':
                mu = np.zeros((self.p,self.K))
                kappa = np.zeros(self.K)
                for k in range(self.K):
                    mu[:,k],kappa[k] = self.M_step_single_component(X=X[X_part==k],beta=np.ones(np.sum(X_part==k)),mu=self.mu[:,k],kappa=self.kappa[k])
                self.mu = mu
                self.kappa = kappa
            elif self.distribution in ['ACG_lowrank','MACG_lowrank']:
                M = np.zeros((self.K,self.p,self.r))
                for k in range(self.K):
                    M[k] = self.M_step_single_component(X=X[X_part==k],beta=np.ones(np.sum(X_part==k)),M=self.M[k],Lambda=None)
                self.M = M
            elif self.distribution in ['ACG_fullrank','MACG_fullrank']:
                Lambda = np.zeros((self.K,self.p,self.p))
                for k in range(self.K):
                    Lambda[k] = self.M_step_single_component(X=X[X_part==k],beta=np.ones(np.sum(X_part==k)),M=None, Lambda=self.Lambda[k])
                self.Lambda=Lambda
            elif self.distribution == 'SingularWishart_lowrank':
                M = np.zeros((self.K,self.p,self.r))
                gamma = np.zeros(self.K)
                for k in range(self.K):
                    M[k],gamma[k] = self.M_step_single_component(Q=X[X_part==k]*np.sqrt(L[X_part==k])[:,None,:],beta=np.ones(np.sum(X_part==k)),M=self.M[k],gamma=self.gamma[k])
                self.M = M
                self.gamma = gamma
            elif self.distribution == 'SingularWishart_fullrank':
                Lambda = np.zeros((self.K,self.p,self.p))
                for k in range(self.K):
                    Lambda[k] = self.M_step_single_component(Q=X[X_part==k]*np.sqrt(L[X_part==k])[:,None,:],beta=np.ones(np.sum(X_part==k)),M=None)
                self.Lambda = Lambda

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

    def update_pi(self,beta):
        self.pi = np.sum(beta,axis=1)/beta.shape[1]

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
        elif self.distribution in ['ACG_lowrank','MACG_lowrank']:
            return {'M':self.M,'pi':self.pi}
        elif self.distribution == 'SingularWishart_lowrank':
            return {'M':self.M,'pi':self.pi,'gamma':self.gamma}
        elif self.distribution in ['ACG_fullrank','MACG_fullrank','SingularWishart_fullrank']:
            return {'Lambda':self.Lambda,'pi':self.pi}
        
    def set_params(self,params):
        self.unpack_params(params)
