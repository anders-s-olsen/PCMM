import numpy as np
import torch
import torch.nn as nn
from src.models_python.WatsonMixtureEM import Watson
from src.models_python.mixture_EM_loop import mixture_EM_loop
from src.models_pytorch.diametrical_clustering_torch import diametrical_clustering_torch, diametrical_clustering_plusplus_torch
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)

class ACG(nn.Module):
    """
    Angular-Central-Gaussian spherical distribution:

    "Tyler 1987 - Statistical analysis for the angular central Gaussian distribution on the sphere"
    """

    def __init__(self, K: int,p: int,rank=None,params=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.K = K
        self.p = torch.tensor(p) #dimensionality
        self.r = rank
        # if rank is None or self.r == self.p:
        #     self.fullrank = True
        # else:
        #     self.fullrank = False
        self.half_p = torch.tensor(p / 2)
        self.logSA = torch.lgamma(self.half_p) - torch.log(torch.tensor(2)) -self.half_p* torch.log(torch.tensor(np.pi))

        self.LogSoftmax = nn.LogSoftmax(dim=0)

        # self.tril_mask = torch.tril_indices(self.p,self.p)
        # self.diag_mask = ((torch.arange(1,self.p+1)**2+torch.arange(1,self.p+1))/2-1).type(torch.LongTensor)
        # self.num_params = int(self.p*(self.p-1)/2+self.p)
        
        if params is not None: # for evaluating likelihood with already-learned parameters
            if torch.is_tensor(params['pi']):
                self.pi = nn.Parameter(params['pi'])
            else:
                self.pi = nn.Parameter(torch.tensor(params['pi']))

            # if self.fullrank:
            #     if torch.is_tensor(params['Lambda']):
            #         self.L_vec = torch.linalg.cholesky(torch.linalg.inv(params['Lambda']))[:,self.tril_mask[0],self.tril_mask[1]]
            #     else:
            #         self.L_vec = torch.linalg.cholesky(torch.linalg.inv(torch.tensor(params['Lambda'])))[:,self.tril_mask[0],self.tril_mask[1]]
            #     self.L_vec = nn.Parameter(self.L_vec)
            # else:
            M_init = params['M']
            if M_init.dim()!=3 or M_init.shape[2]!=self.r: # add extra columns
                if M_init.dim()==2:
                    num_missing = self.r-1
                else:
                    num_missing = self.r-M_init.shape[2]

                M_extra = torch.randn(self.K,self.p,num_missing)
                self.M = nn.Parameter(torch.cat([M_init,M_extra],dim=2))
            else: 
                self.M = M_init
            

    def get_params(self):
        # if self.fullrank is True:
        #     L_tri_inv = torch.zeros(self.K,self.p,self.p,device=self.device)
        #     L_tri_inv[:,self.tril_mask[0],self.tril_mask[1]] = self.L_vec.data
        #     return {'L_tri_inv':L_tri_inv,'pi':self.pi.data} # should be L = inv(S_tri_inv@S_tri_inv.T)
        # else:
        return {'M':self.M.data,'pi':self.pi.data} #should be normalized: L=MM^T+I and then L = p*L/trace(L)

    def initialize(self,X=None,init=None,tol=None):

        self.pi = nn.Parameter(torch.ones(self.K,device=self.device)/self.K)
        if init == 'no':
            return
        if init is not None and init !='unif' and init!='uniform':
            if init == '++' or init == 'plusplus' or init == 'diametrical_clustering_plusplus':
                mu = diametrical_clustering_plusplus_torch(X=X,K=self.K)
            elif init == 'dc' or init == 'diametrical_clustering':
                mu = diametrical_clustering_torch(X=X,K=self.K,max_iter=100000,num_repl=5,init='++',tol=tol)
            elif init == 'WMM' or init == 'Watson' or init == 'W' or init == 'watson':
                W = Watson(K=self.K,p=self.p)
                params,_,_,_ = mixture_EM_loop(W,X,init='dc')
                mu = params['mu']
                self.pi = nn.Parameter(params['pi'])
            # if self.fullrank is True:
            #     self.L_vec = torch.zeros((self.K,self.num_params)).to(self.device)
            #     for k in range(self.K):
            #         self.L_vec[k] = torch.linalg.cholesky(torch.outer(mu[:,k],mu[:,k])+torch.eye(self.p))[self.tril_mask[0],self.tril_mask[1]]
            #     self.L_vec = nn.Parameter(self.L_vec)
            # else:
            self.M = torch.rand((self.K,self.p,self.r)).to(self.device)
            for k in range(self.K):
                self.M[k,:,0] = mu[:,k] #initialize only the first of the rank D columns this way, the rest uniform
            self.M = nn.Parameter(self.M)
        elif init =='unif' or init=='uniform' or init is None:
            # if self.fullrank is True:
            #     self.L_vec = nn.Parameter(torch.rand((self.K,self.num_params)).to(self.device))
            # else:
            self.M = nn.Parameter(torch.rand((self.K,self.p,self.r)).to(self.device))
                

    def log_pdf(self,X):

        # if self.fullrank is True:
        #     L_tri_inv = torch.zeros(self.K,self.p,self.p,device=self.device,dtype=torch.double)
        #     L_tri_inv[:, self.tril_mask[0], self.tril_mask[1]] = self.L_vec
        #     B = X[None,:,:] @ L_tri_inv
        #     pdf = torch.sum(B * B, dim=2)
        #     log_det_L = -2 * torch.sum(torch.log(torch.abs(self.L_vec[:,self.diag_mask])),dim=1)
        # else:

        Lambda = torch.eye(self.r) + torch.swapaxes(self.M,-2,-1)@self.M
        log_det_L = torch.logdet(Lambda)
        B = X[None,:,:]@self.M
        pdf = 1-torch.sum(B@torch.linalg.inv(Lambda)*B,dim=2) #check

        # minus log_det_L instead of + log_det_A_inv
        log_acg_pdf = self.logSA - 0.5 * log_det_L[:,None] - self.half_p * torch.log(pdf)
        return log_acg_pdf
    
    def log_density(self,X):
        return self.log_pdf(X)+self.LogSoftmax(self.pi)[:,None]

    def log_likelihood(self, X):
        density = self.log_density(X)
        logsum_density = torch.logsumexp(density, dim=0)
        loglik = torch.sum(logsum_density)
        return loglik
    
    def test_log_likelihood(self, X): #without constraints (assume pi sum to one)
        density = self.log_pdf(X)+torch.log(self.pi)[:,None]
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
    # test that the code works
    import matplotlib.pyplot as plt

    dim = 3
    ACG = ACG(p=dim,D=2)
    
    #ACG_pdf = lambda phi: float(torch.exp(ACG(torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float))))
    #acg_result = scipy.integrate.quad(ACG_pdf, 0., 2*np.pi)

    X = torch.randn(6, dim)

    out = ACG(X)
    print(out)
    # # ACG.L_under_diag = nn.Parameter(torch.ones(2,2))
    # # ACG.L_diag = nn.Parameter(torch.tensor([21.,2.5]))
    # phi = torch.arange(0, 2*np.pi, 0.001)
    # phi_arr = np.array(phi)
    # x = torch.column_stack((torch.cos(phi),torch.sin(phi)))
    #
    # points = torch.exp(ACG(x))
    # props = np.array(points.squeeze().detach())
    #
    # ax = plt.axes(projection='3d')
    # ax.plot(np.cos(phi_arr), np.sin(phi_arr), 0, 'gray') # ground line reference
    # ax.scatter(np.cos(phi_arr), np.sin(phi_arr), props, c=props, cmap='viridis', linewidth=0.5)
    #
    # ax.view_init(30, 135)
    # plt.show()
    # plt.scatter(phi,props, s=3)
    # plt.show()
