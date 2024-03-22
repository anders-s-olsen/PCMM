import torch
import torch.nn as nn
from src.DMM_pytorch.diametrical_clustering_torch import diametrical_clustering_torch, diametrical_clustering_plusplus_torch, grassmannian_clustering_gruber2006_torch
from src.DMM_pytorch.run_single_comp import run_single_comp
class DMMPyTorchBaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.LogSoftmax_pi = nn.LogSoftmax(dim=0)
        self.LogSoftmax_T = nn.LogSoftmax(dim=1)

    def unpack_params(self,params):
        # check if the any one of the elements in the dictionary is a tensor
        
        if not torch.is_tensor(params[list(params.keys())[0]]):
            raise ValueError('Input parameters must be torch tensors')
        
        # mixture or HMM settings
        if 'pi' in params:
            self.pi = nn.Parameter(params['pi'])
        else:
            if self.K==1:
                self.pi = torch.tensor(1)
            else:
                self.pi = nn.Parameter(torch.tensor(1/self.K).repeat(self.K))
        if self.HMM:
            if 'T' in params:
                self.T = nn.Parameter(params['T'])
            else:
                self.T = nn.Parameter(torch.tensor(1/self.K).repeat(self.K,self.K))
        
        # distribution-specific settings
        if self.distribution == 'Watson':
            self.mu = nn.Parameter(params['mu'])
            self.kappa = nn.Parameter(params['kappa'])
        elif self.distribution == 'ACG' or self.distribution == 'MACG':
            M_init = params['M']
                
            if M_init.dim()!=3 or M_init.shape[2]!=self.r: # add extra columns
                if M_init.dim()==2:
                    num_missing = self.r-1
                else:
                    num_missing = self.r-M_init.shape[2]

                M_extra = torch.randn(self.K,M_init.shape[1],num_missing)
                self.M = nn.Parameter(torch.cat([M_init,M_extra],dim=2))
            else:
                self.M = nn.Parameter(M_init)
        else:
            raise ValueError('Invalid distribution')



    def initialize(self,X,init_method,model2=None):
        assert init_method in ['uniform','unif',
                               '++','plusplus','diametrical_clustering_plusplus','dc++',
                               '++_seg','plusplus_seg','diametrical_clustering_plusplus_seg','dc++_seg',
                               'dc','diametrical_clustering',
                               'dc_seg','diametrical_clustering_seg',
                               'Grassmann','Grassmann_clustering',
                               'Grassmann_seg','Grassmann_clustering_seg']
        assert self.distribution in ['Watson','ACG','ACG','MACG','MACG']

        # strategy: 
        # 1) if uniform, initialize with uniform random values and return
        # 2) if not, compute mu or C using dc++, dc, gr or gr++
        # 3) if not seg use those values to initialize the model
        # 4) if seg, use the partition and run the model a single time using the partition

        # initialize pi as uniform, will be overwritten by 'seg' methods
        self.pi = self.pi = nn.Parameter((1/self.K).repeat(self.K))

        # for watson, always initialize kappa as ones, will be overwritten by 'seg' methods
        if self.distribution == 'Watson':
            self.kappa = nn.Parameter(torch.ones(self.K))

        if init_method in ['uniform','unif']:
            if self.distribution == 'Watson':
                mu = torch.rand(self.p,self.K)
                self.mu = nn.Parameter(mu/torch.linalg.vector_norm(mu,dim=0))
            elif self.distribution in ['ACG', 'MACG']:
                self.M = nn.Parameter(torch.rand(self.K,self.p,self.r))
            return
        elif init_method in ['++','plusplus','diametrical_clustering_plusplus','dc++','++_seg','plusplus_seg','diametrical_clustering_plusplus_seg','dc++_seg','dc','diametrical_clustering','dc_seg','diametrical_clustering_seg']:
            if X.ndim==3:
                X2 = X[:,:,0]
            else:
                X2 = X.clone()
            
            # if '++' or 'plusplus' in init_method
            if init_method in ['++','plusplus','diametrical_clustering_plusplus','dc++','++_seg','plusplus_seg','diametrical_clustering_plusplus_seg','dc++_seg']:
                mu = diametrical_clustering_plusplus_torch(X=X2,K=self.K)
            elif init_method in ['dc','diametrical_clustering','dc_seg','diametrical_clustering_seg']:
                mu = diametrical_clustering_torch(X=X2,K=self.K,max_iter=100000,num_repl=1,init='++')
            
            if self.distribution == 'Watson':
                self.mu = nn.Parameter(mu)
            elif self.distribution in ['ACG','MACG']:
                self.M = torch.rand(self.K,self.p,self.r)
                for k in range(self.K):
                    self.M[k,:,0] = mu[:,k]
                self.M = nn.Parameter(self.M)

            # if segmentation methods, use mu to segment the data and later estimate single-component models
            if init_method in ['++_seg','plusplus_seg','diametrical_clustering_plusplus_seg','dc++_seg','dc_seg','diametrical_clustering_seg']:
                sim = (X2@mu)**2           
                X_part = torch.argmax(sim,dim=1) 
            
        elif init_method in ['Grassmann','Grassmann_clustering','Grassmann_seg','Grassmann_clustering_seg']:
            if X.ndim!=3:
                raise ValueError('Grassmann methods are only implemented for 3D data')
            
            C = grassmannian_clustering_gruber2006_torch(X=X,K=self.K,max_iter=100000,num_repl=1)
            
            self.M = torch.rand(self.K,self.p,self.r)
            for k in range(self.K):
                self.M[k,:,0] = C[k,:,0]
            self.M = nn.Parameter(self.M)

            if init_method in ['Grassmann_seg','Grassmann_clustering_seg']:
                XXt = X@torch.swapaxes(X,-2,-1)
                CCt = C@torch.swapaxes(C,-2,-1)
                dis = 1/torch.sqrt(torch.tensor(2))*torch.linalg.matrix_norm(XXt[:,None]-CCt[None])
                X_part = torch.argmin(dis,axis=1)
        else:
            raise ValueError('Invalid init_method')
            
        if init_method in ['++_seg','plusplus_seg','diametrical_clustering_plusplus_seg','dc++_seg','dc_seg','diametrical_clustering_seg','Grassmann_seg','Grassmann_clustering_seg']:
            print('Running single component models as initialization')
            self.pi = nn.Parameter(torch.bincount(X_part)/X_part.shape[0])
            # run a single-component model on each segment
            if self.distribution=='Watson':
                mu = torch.zeros(self.p,self.K)
                kappa = torch.zeros(self.K)
                for k in range(self.K):
                    model2.unpack_params({'mu':self.mu[:,k],'kappa':torch.ones(1),'pi':torch.zeros(1)})
                    params,_,_ = run_single_comp(model2,X[X_part==k])
                    mu[:,k] = params['mu'][:,0]
                    kappa[k] = params['kappa']
                self.mu = nn.Parameter(mu)
                self.kappa = nn.Parameter(kappa)
            elif self.distribution in ['ACG','MACG']:
                M = torch.zeros(self.K,self.p,self.r)
                for k in range(self.K):
                    model2.unpack_params({'M':self.M[k][None],'pi':torch.zeros(1)})
                    params,_,_ = run_single_comp(model2,X[X_part==k],rank=self.r)
                    M[k] = params['M']
                self.M = nn.Parameter(M)

    # here the log_pdf should be formatted as KxN
    def MM_log_likelihood(self,log_pdf):
        # if log_pdf.ndim!=2:
        #     raise ValueError("log_pdf should be KxN, where N is concatenated across all subjects/batches")
        log_density = log_pdf+self.LogSoftmax_pi(self.pi)[:,None] #each pdf gets "multiplied" with the weight
        logsum_density = torch.logsumexp(log_density,dim=0) #sum over the K components
        log_likelihood = torch.sum(logsum_density) #sum over the N samples
        return log_likelihood

    # Here the log_pdf should be formatted as KxN
    def HMM_log_likelihood(self,log_pdf):
        # if log_pdf.ndim!=3:
        #     raise ValueError("log_pdf should be BxKxN")
        K,N = log_pdf.shape
        log_T = self.LogSoftmax_T(self.T) # size KxK
        log_pi = self.LogSoftmax_pi(self.pi) #size K

        # log_alpha = torch.zeros_like(log_pdf)
        log_alpha = torch.zeros(N,K)

        #initialize the first alpha for time t=0
        log_alpha[0,:] = log_pdf[:,0]+log_pi

        #recursion for time t=1 and onwards
        for t in range(1,N):
            log_alpha[t,:] = log_pdf[:,t]+torch.logsumexp(log_alpha[t-1,:]+log_T,dim=-1)

        # Logsum for states N for each time t
        log_t = torch.logsumexp(log_alpha,dim=-1)

        # Retrieve alpha for the last time t
        log_probability = log_t[-1]

        return log_probability
    
    def forward(self, X):
        log_pdf = self.log_pdf(X)
        if self.K==1:
            return torch.sum(log_pdf)
        else:
            if self.HMM:
                return self.HMM_log_likelihood(log_pdf)
            else:
                return self.MM_log_likelihood(log_pdf)
        
    def test_log_likelihood(self,X):
        with torch.no_grad():
            return self.forward(X)
        
    def posterior_MM(self,log_pdf):
        log_density = log_pdf+self.LogSoftmax_pi(self.pi)[:,None]
        logsum_density = torch.logsumexp(log_density,dim=0)
        return torch.exp(log_density-logsum_density)

    def viterbi(self,log_pdf):
        K,N = log_pdf.shape
        log_T = self.LogSoftmax_T(self.T) # size KxK
        log_pi = self.LogSoftmax_pi(self.pi) #size K
        
        log_delta = torch.zeros(N,K)
        log_psi = torch.zeros(N,K,dtype=torch.int32)

        log_delta[0,:] = log_pdf[:,0]+log_pi

        for t in range(1,N):
            temp = log_delta[t-1,:]+log_T+log_pdf[:,t]
            log_delta[t,:],log_psi[t,:] = torch.max(temp,dim=1)

        Z_T_prob, Z_T = torch.max(log_delta[-1,:],dim=0)
        Z_path = torch.zeros(N,dtype=torch.int32)
        Z_path[-1] = Z_T

        for t in range(N-2,-1,-1):
            Z_path[t] = log_psi[t+1,Z_T]
            Z_T = Z_path[t]

        Z_path2 = torch.zeros(K,N,dtype=torch.bool)
        for t in range(N):
            Z_path2[Z_path[t],t] = True

        return Z_path2


    def viterbi_multi(self,log_pdf): #for multisubject
        raise NotImplementedError("Viterbi() method not implemented")
        if log_pdf.ndim!=3:
            raise ValueError("log_pdf should be BxKxN")
        B,K,N = log_pdf.shape
        log_T = self.LogSoftmax_T(self.T) # size KxK
        log_pi = self.LogSoftmax_pi(self.pi) #size K

        log_delta = torch.zeros_like(log_pdf)
        log_psi = torch.zeros_like(log_pdf,dtype=torch.int32) #should be BxN no?
        Z_path = torch.zeros((B,N),dtype=torch.int32)

        # init at t=0
        log_delta[:,:,0] = log_pdf[:,:,0]+log_pi

        # recursion for time t=1 and onwards
        for t in range(1,N):
            temp = log_delta[:,:,t-1].unsqueeze(1)+log_T + log_pdf[:,:,t].unsqueeze(1)
            log_delta[:,:,t],log_psi[:,:,t] = torch.max(temp,dim=2)
            # log_delta += log_pdf[:,:,t]
        
        # backtrack
        T_log_delta = log_delta[:,:,-1]
        Z_T_prob, Z_T = torch.max(T_log_delta,dim=1)
        Z_path[:,-1] = Z_T

        for t in range(N-2,-1,-1):
            Z_path[:,t] = log_psi[:,Z_T,t+1]
        
        return Z_path
    
    def posterior(self,X):
        with torch.no_grad():
            log_pdf = self.log_pdf(X)
            if self.HMM:
                return self.viterbi(log_pdf)#self.posterior_MM(log_pdf), 
            else:
                return self.posterior_MM(log_pdf)
    
    def get_params(self):
        pi = self.LogSoftmax_pi(self.pi.data).detach()
        if self.distribution == 'Watson':
            mu = nn.functional.normalize(self.mu, dim=0).detach()
            return {'mu':mu,'kappa':self.kappa.data,'pi':pi}
        elif self.distribution == 'ACG' or self.distribution == 'MACG':
            return {'M':self.M.detach(),'pi':pi}
        
    def set_params(self,params):
        self.unpack_params(params)
        
    def sample(self, num_samples):
        raise NotImplementedError("Subclasses must implement sample() method")
