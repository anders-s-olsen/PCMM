import numpy as np
import torch
import torch.nn as nn
from src.models_python.WatsonMixtureEM import Watson
from src.models_python.mixture_EM_loop import mixture_EM_loop
from src.models_python.diametrical_clustering_torch import diametrical_clustering_torch, diametrical_clustering_plusplus_torch

from scipy.special import gamma

#device = 'cpu'
class MACG(nn.Module):
    """
    Angular-Central-Gaussian spherical distribution:

    "Tyler 1987 - Statistical analysis for the angular central Gaussian distribution on the sphere"
    """

    def __init__(self, K: int,p: int,q:int,rank=None,params=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.K = K
        self.p = torch.tensor(p) #dimensionality
        self.D = torch.tensor(rank) 
        self.q = q
        if rank is None or self.D == self.p:
            self.fullrank = True
        else:
            self.fullrank = False
        self.c = torch.tensor(p / 2)
        self.gamma_k = torch.tensor(np.pi)**(self.q*(self.q-1)/4)
        for i in range(self.q):
            self.gamma_k *= torch.tensor(gamma(self.c-i/2)) # should be (i-1)/2 but python is zero-indexed :(
        self.logSA_Stiefel = torch.log(self.gamma_k)-q*torch.log(torch.tensor(2))-self.q*self.p/2*torch.log(torch.tensor(np.pi))

        self.LogSoftmax = nn.LogSoftmax(dim=0)

        self.tril_indices = torch.tril_indices(self.p,self.p)
        self.diag_indices = torch.zeros(self.p).type(torch.LongTensor)
        self.num_params = int(self.p*(self.p-1)/2+self.p)
        for i in range(1,self.p+1):   
            self.diag_indices[i-1] = ((i**2+i)/2)-1
        
        if params is not None: # for evaluating likelihood with already-learned parameters
            self.pi = params['pi']
            if self.fullrank: #check if this works!!
                self.S_vec = torch.zeros(self.K,self.num_params,device=self.device,dtype=torch.double)
                for k in range(self.K):
                    self.S_vec[k] = torch.linalg.cholesky(torch.linalg.inv(params['Sigma'][k]))[self.tril_indices[0],self.tril_indices[1]]
            else:
                M_init = params['M']
                if M_init.dim()!=3 or M_init.shape[2]!=self.D: # add extra columns
                    if M_init.dim()==2:
                        num_missing = self.D-1
                    else:
                        num_missing = self.D-M_init.shape[2]

                    M_extra = torch.randn(self.K,self.p,num_missing,dtype=torch.double)
                    self.M = nn.Parameter(torch.cat([M_init,M_extra],dim=2))
            

    def get_params(self):
        if self.fullrank is True:
            S_tri_inv = torch.zeros(self.K,self.p,self.p,device=self.device,dtype=torch.double)
            for k in range(self.K):
                S_tri_inv[k,self.tril_indices[0],self.tril_indices[1]] = self.S_vec[k].data
            return {'S_tri_inv':S_tri_inv,'pi':self.pi.data} # should be L = inv(S_tri_inv@S_tri_inv.T)
        else:
            return {'M':self.M.data,'pi':self.pi.data} #should be normalized: L=MM^T+I and then L = p*L/trace(L)

    def initialize(self,X=None,init=None):

        self.pi = nn.Parameter(torch.ones(self.K,device=self.device)/self.K)
        if self.fullrank is True:
            # only initialize the cholesky formulation as random, since this one is susceptible to singularity by rank-one methods
            self.S_vec = nn.Parameter(torch.randn((self.K,self.num_params),dtype=torch.double).to(self.device))
        else:
            if init is None or init=='uniform' or init=='unif':
                self.M = nn.Parameter(torch.rand((self.K,self.p,self.D),dtype=torch.double).to(self.device))
            else:
                if init == '++' or init == 'plusplus' or init == 'diametrical_clustering_plusplus':
                    mu = diametrical_clustering_plusplus_torch(X=X,K=self.K)
                elif init == 'dc' or init == 'diametrical_clustering':
                    mu,_,_ = diametrical_clustering_torch(X=X,K=self.K,max_iter=100000,num_repl=5,init='++')
                elif init == 'WMM' or init == 'Watson' or init == 'W' or init == 'watson':
                    W = Watson(K=self.K,p=self.p)
                    params,_,_,_ = mixture_EM_loop(W,X,init='dc')
                    mu = params['mu']
                    self.pi = nn.Parameter(params['pi'])
                self.M = nn.Parameter(torch.rand((self.K,self.p,self.D),dtype=torch.double).to(self.device))
                for k in range(self.K):
                    self.M[k,:,0] = mu[:,k] #initialize only the first of the rank D columns this way, the rest uniform
                

    def log_determinant_L(self,L):
        log_det_L = torch.log(torch.linalg.det(L))
        
        return log_det_L
    
    def log_pdf(self,X):

        if self.fullrank is True:
            S_tri_inv = torch.zeros(self.K,self.p,self.p,device=self.device,dtype=torch.double)
            pdf = torch.zeros((self.K,X.shape[0]))
            for k in range(self.K):
                S_tri_inv[k,self.tril_indices[0],self.tril_indices[1]] = self.S_vec[k]
                pdf[k] = torch.linalg.det(torch.swapaxes(X,-2,-1)@S_tri_inv[k]@S_tri_inv[k].T@X) ### revisit
            log_det_L = -2 * torch.sum(torch.log(torch.abs(self.S_vec[:,self.diag_indices])),dim=1)
            
        else:
            Sigma = torch.zeros(self.K,self.D,self.D)
            for k in range(self.K):
                self.Sigma[k] = self.p*Sigma[k]/torch.trace(Sigma[k]) #trace-normalize, check if this is also invariant
                pdf[k] = np.linalg.det(np.swapaxes(X,-2,-1)@np.linalg.inv(self.Sigma[k])@X)
            log_det_L = self.log_determinant_L(self.Sigma)

        # minus log_det_L instead of + log_det_A_inv
        log_acg_pdf = self.logSA_Stiefel - self.q/2 * log_det_L[:,None] - self.c * torch.log(pdf)
        return log_acg_pdf
    
    def log_density(self,X):
        return self.log_pdf(X)+self.LogSoftmax(self.pi)[:,None]

    def log_likelihood(self, X):
        density = self.log_density(X)
        logsum_density = torch.logsumexp(density, dim=0)
        loglik = torch.sum(logsum_density)
        return loglik

    def forward(self, X):
        return self.log_likelihood(X)
    
    def posterior(self,X):
        density = self.log_density(X)
        logsum_density = torch.logsumexp(density, dim=0)
        return torch.exp(density-logsum_density)
    
    def __repr__(self):
        return 'ACG'


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    K = np.array(2)
    
    p = np.array(3)
    MACG = MACG(K=K,p=3,q=2,rank=p)
    data = np.loadtxt('data/synthetic/synth_data_MACG.csv',delimiter=',')
    data2 = np.zeros((1000,p,2))
    data2[:,:,0] = data[np.arange(2000,step=2),:]
    data2[:,:,1] = data[np.arange(2000,step=2)+1,:]
    data2 = torch.tensor(data2)
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