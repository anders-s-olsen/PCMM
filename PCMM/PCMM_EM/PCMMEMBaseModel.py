import numpy as np
from PCMM.riemannian_clustering import diametrical_clustering, plusplus_initialization, grassmann_clustering, weighted_grassmann_clustering
from scipy.cluster.vq import kmeans2

class PCMMEMBaseModel():

    def __init__(self):
        super().__init__()
    
    def unpack_params(self,params):

        # distribution-specific settings
        if 'Watson' in self.distribution:
            self.mu = params['mu']
            self.kappa = params['kappa']
        elif 'lowrank' in self.distribution:
            self.M = params['M']
            if self.M.ndim==2 and self.K==1:
                self.M = self.M[None,:,:]
        elif 'fullrank' in self.distribution:
            self.Lambda = params['Lambda']
        else:
            raise ValueError('Invalid distribution')
        
        if self.distribution in ['SingularWishart_lowrank','Normal_lowrank','Complex_Normal_lowrank']:
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
        if V.shape[1]>=self.p:
            epsilon = np.sum(S[r:])/(self.p-r)
        else:
            epsilon = np.sum(S[r:])/(V.shape[1]-r)
        M = U[:,:r]@np.diag(np.sqrt(((S[:r]-epsilon)/epsilon)))

        if 'Normal' in self.distribution and self.r==1:
            return M,epsilon#/V.shape[1]        
        else:
            return M,epsilon/V.shape[1]        
    
    def init_M_svd_given_M_init(self,X,M_init,beta=None,gamma=None):
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

        # initialize remainder using the svd of the residual of X after subtracting the projection on M_init
        if 'Complex' in self.distribution:
            M = np.zeros((self.K,self.p,self.r),dtype=complex)
        else:
            M = np.zeros((self.K,self.p,self.r))
        
        for k in range(self.K):
            if X.ndim==2:
                V = (beta[k][:,None]*X).T
            else:
                V = np.reshape(np.swapaxes(beta[k][:,None,None]*X,0,1),(self.p,self.q*X.shape[0]))
                
            # projection matrix of M
            if self.distribution in ['ACG_lowrank','Complex_ACG_lowrank','MACG_lowrank']:
                gamma = 1/(1+np.linalg.norm(M_init[k],'fro')**2/self.p)
                M_init[k] = np.sqrt(gamma)*M_init[k]
                M_proj = M_init[k]@np.linalg.inv(M_init[k].T.conj()@M_init[k]+np.eye(M_init[k].shape[1]))@M_init[k].T.conj()
            elif self.distribution in ['SingularWishart_lowrank','Normal_lowrank','Complex_Normal_lowrank']:
                M_proj = M_init[k]@np.linalg.inv(M_init[k].T.conj()@M_init[k]+self.gamma[k]*np.eye(M_init[k].shape[1]))@M_init[k].T.conj()
            # M_proj = M_init[k]@np.linalg.inv(M_init[k].T@M_init[k])@M_init[k].T
            V_residual = V - M_proj@V
            M_extra,_ = self.init_M_svd(V_residual,r=num_missing)
            M[k] = np.concatenate([M_init[k],M_extra],axis=-1)
        return M

    def initialize(self,X,init_method):
        assert init_method in ['uniform','unif',
                               'diametrical_clustering_plusplus','dc++','diametrical_clustering_plusplus_seg','dc++_seg',
                               'grassmann_clustering_plusplus','gc++','grassmann_clustering_plusplus_seg','gc++_seg',
                               'weighted_grassmann_clustering_plusplus','wgc++','weighted_grassmann_clustering_plusplus_seg','wgc++_seg',
                               'dc','diametrical_clustering','dc_seg','diametrical_clustering_seg',
                               'gc','grassmann_clustering','gc_seg','grassmann_clustering_seg',
                               'wgc','weighted_grassmann_clustering','wgc_seg','weighted_grassmann_clustering_seg',
                               'euclidean','euclidean_seg']
        assert self.distribution in ['Watson','Complex_Watson',
                                     'ACG_lowrank','ACG_fullrank',
                                     'Complex_ACG_lowrank','Complex_ACG_fullrank',
                                     'MACG_lowrank','MACG_fullrank',
                                     'SingularWishart_lowrank','SingularWishart_fullrank',
                                     'Normal_lowrank','Normal_fullrank',
                                     'Complex_Normal_lowrank','Complex_Normal_fullrank']

        # for watson, always initialize kappa as ones, will be overwritten by 'seg' methods
        if 'Watson' in self.distribution:
            self.kappa = np.ones(self.K)
        elif 'SingularWishart_lowrank' in self.distribution or 'Normal_lowrank' in self.distribution:
            self.gamma = np.ones(self.K)

        if init_method in ['uniform','unif']:
            print('Initializing parameters using the uniform distribution')
            self.pi = np.array([1/self.K]*self.K)
            if 'Watson' in self.distribution:
                if X.dtype==complex:
                    mu = np.random.uniform(size=(self.p,self.K))+1j*np.random.uniform(size=(self.p,self.K))
                else:
                    mu = np.random.uniform(size=(self.p,self.K))
                self.mu = mu / np.linalg.norm(mu,axis=0)
            else:
                if X.dtype==complex:
                    M = np.random.uniform(size=(self.K,self.p,self.r))+1j*np.random.uniform(size=(self.K,self.p,self.r))
                else:
                    M = np.random.uniform(size=(self.K,self.p,self.r))
                if 'fullrank' in self.distribution:
                    self.Lambda = np.zeros((self.K,self.p,self.p),dtype=M.dtype)
                    for k in range(self.K):
                        self.Lambda[k] = M[k]@M[k].T.conj()+np.eye(self.p)
                        self.Lambda[k] = self.p*self.Lambda[k]/np.trace(self.Lambda[k])
                else:
                    self.M = M
            return           
        elif init_method in ['euclidean','euclidean_seg','diametrical_clustering_plusplus','dc++','dc++_seg','diametrical_clustering_plusplus_seg','dc','diametrical_clustering','dc_seg','diametrical_clustering_seg']:
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
            elif init_method in ['euclidean','euclidean_seg']:
                X2[(X2.real>0).sum(axis=1)>self.p/2] = -X2[(X2.real>0).sum(axis=1)>self.p/2]
                for l in range(100): #try until we get a valid clustering......................
                    mu,X_part = kmeans2(X2,k=self.K,minit='++')
                    pi = np.bincount(X_part)/X_part.shape[0]
                    if not np.any(pi<1/(self.K*4)):
                        break
                mu = mu.T
            
            if 'Watson' in self.distribution:
                print('Initializing mu based on the clustering centroid')
                self.mu = mu
            elif self.distribution in ['ACG_lowrank','MACG_lowrank','Complex_ACG_lowrank','Normal_lowrank','Complex_Normal_lowrank']:
                print('Initializing M based on a lowrank-svd of the input data partitioned acc to the clustering')
                self.M = np.zeros((self.K,self.p,self.r),dtype=X.dtype)
                gamma = np.zeros(self.K)
                for k in range(self.K):
                    self.M[k],gamma[k] = self.init_M_svd(X[X_part==k].T,self.r)
                if 'Normal' in self.distribution:
                    self.gamma = gamma
            elif self.distribution in ['MACG_lowrank','SingularWishart_lowrank']:
                print('Initializing M based on a lowrank-svd of the input data partitioned acc to the clustering')
                self.M = np.zeros((self.K,self.p,self.r),dtype=X.dtype)
                for k in range(self.K):
                    self.M[k],gamma = self.init_M_svd(np.reshape(np.swapaxes(X[X_part==k],0,1),(self.p,np.sum(X_part==k)*self.q)),self.r)
                if self.distribution == 'SingularWishart_lowrank':
                    self.gamma = gamma
            elif 'fullrank' in self.distribution:
                print('Initializing Lambda based on the clustering centroids')
                self.Lambda = np.zeros((self.K,self.p,self.p),dtype=X.dtype)
                for k in range(self.K):
                    self.Lambda[k] = np.outer(mu[:,k],mu[:,k])+np.eye(self.p)
                    if self.distribution in ['ACG_fullrank','MACG_fullrank','Complex_ACG_fullrank']:
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
            
            if 'lowrank' in self.distribution:
                print('Initializing M based on a lowrank-svd of the input data partitioned acc to the clustering')
                self.M = np.zeros((self.K,self.p,self.r),dtype=X.dtype)
                gamma = np.zeros(self.K)
                for k in range(self.K):
                    self.M[k],gamma[k] = self.init_M_svd(np.reshape(np.swapaxes(X[X_part==k],0,1),(self.p,np.sum(X_part==k)*self.q)),self.r)
                if 'SingularWishart' in self.distribution:
                    self.gamma = gamma
            elif 'fullrank' in self.distribution:
                print('Initializing Lambda based on the clustering centroids')
                self.Lambda = np.zeros((self.K,self.p,self.p),dtype=X.dtype)
                for k in range(self.K):
                    self.Lambda[k] = C[k]@C[k].T.conj()+np.eye(self.p)
                    if self.distribution in ['ACG_fullrank','MACG_fullrank','Complex_ACG_fullrank']:
                        self.Lambda[k] = self.p*self.Lambda[k]/np.trace(self.Lambda[k])

        elif init_method in ['weighted_grassmann_clustering','weighted_grassmann_clustering_seg','wgc','wgc_seg','weighted_grassmann_clustering_plusplus','wgc++','weighted_grassmann_clustering_plusplus_seg','wgc++_seg']:
            if X.ndim!=3:
                raise ValueError('Grassmann methods are only implemented for 3D data')
            
            if init_method in ['weighted_grassmann_clustering_plusplus','wgc++','weighted_grassmann_clustering_plusplus_seg','wgc++_seg']:
                print('Running weighted grassmann clustering ++ initialization')
                C,C_weights,X_part,_ = plusplus_initialization(X=X,K=self.K,dist='weighted_grassmann')
            elif init_method in ['wgc','weighted_grassmann_clustering','wgc_seg','weighted_grassmann_clustering_seg']:
                print('Running weighted grassmann clustering initialization')
                C,C_weights,X_part,_ = weighted_grassmann_clustering(X=X,K=self.K,max_iter=100000,num_repl=1)
            
            if 'lowrank' in self.distribution:
                print('Initializing M based on a lowrank-svd of the input data partitioned acc to the clustering')
                self.M = np.zeros((self.K,self.p,self.r),dtype=X.dtype)
                gamma = np.zeros(self.K)
                for k in range(self.K):
                    self.M[k],gamma[k] = self.init_M_svd(np.reshape(np.swapaxes(X[X_part==k],0,1),(self.p,np.sum(X_part==k)*self.q)),self.r)
                if 'SingularWishart' in self.distribution:
                    self.gamma = gamma
            elif 'fullrank' in self.distribution:
                print('Initializing Lambda based on the clustering centroids')
                self.Lambda = np.zeros((self.K,self.p,self.p),dtype=X.dtype)
                for k in range(self.K):
                    self.Lambda[k] = C[k]@np.diag(C_weights[k])@C[k].T.conj()+np.eye(self.p)
                
        else:
            raise ValueError('Invalid init_method')
        
        self.pi = np.bincount(X_part)/X_part.shape[0]
            
        if '_seg' in init_method:
            print('Estimating single component models as initialization')
            # run a single-component model on each segment
            if 'Watson' in self.distribution:
                for k in range(self.K):
                    self.mu[:,k],self.kappa[k] = self.M_step_single_component(X[X_part==k],beta=np.ones(np.sum(X_part==k)),mu=self.mu[:,k],kappa=self.kappa[k])
            elif self.distribution in ['ACG_lowrank','MACG_lowrank','Complex_ACG_lowrank']:
                for k in range(self.K):
                    self.M[k] = self.M_step_single_component(X[X_part==k],beta=np.ones(np.sum(X_part==k)),M=self.M[k],Lambda=None)
            elif self.distribution in ['ACG_fullrank','MACG_fullrank','Complex_ACG_fullrank']:
                for k in range(self.K):
                    self.Lambda[k] = self.M_step_single_component(X[X_part==k],beta=np.ones(np.sum(X_part==k)),M=None, Lambda=self.Lambda[k])
            elif self.distribution in ['SingularWishart_lowrank','Normal_lowrank','Complex_Normal_lowrank']:
                for k in range(self.K):
                    self.M[k],self.gamma[k] = self.M_step_single_component(X[X_part==k],beta=np.ones(np.sum(X_part==k)),M=self.M[k],gamma=self.gamma[k])
            elif self.distribution in ['SingularWishart_fullrank','Normal_fullrank','Complex_Normal_fullrank']:
                for k in range(self.K):
                    self.Lambda[k] = self.M_step_single_component(X[X_part==k],beta=np.ones(np.sum(X_part==k)),M=None)

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

    def log_likelihood(self, X, return_samplewise_likelihood=False):
        log_pdf = self.log_pdf(X)
        if self.K==1:
            if return_samplewise_likelihood:
                return np.sum(log_pdf), log_pdf[0]
            else:
                return np.sum(log_pdf)
        else:
            log_likelihood, self.log_density, self.logsum_density = self.MM_log_likelihood(log_pdf)
            if return_samplewise_likelihood:
                return log_likelihood, self.logsum_density
            else:
                return log_likelihood
        
    def test_log_likelihood(self,X):
        return self.log_likelihood(X,return_samplewise_likelihood=True)

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
        if 'Watson' in self.distribution:
            return {'mu':self.mu,'kappa':self.kappa,'pi':self.pi}
        elif self.distribution in ['ACG_lowrank','Complex_ACG_lowrank','MACG_lowrank']:
            return {'M':self.M,'pi':self.pi}
        elif self.distribution in ['SingularWishart_lowrank','Normal_lowrank','Complex_Normal_lowrank']:
            return {'M':self.M,'pi':self.pi,'gamma':self.gamma}
        elif 'fullrank' in self.distribution:
            return {'Lambda':self.Lambda,'pi':self.pi}
        
    def set_params(self,params):
        self.unpack_params(params)
