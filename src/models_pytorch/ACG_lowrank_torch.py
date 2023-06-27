import numpy as np
import torch
import torch.nn as nn

#from scipy.special import gamma

#device = 'cpu'
class ACG(nn.Module):
    """
    Angular-Central-Gaussian spherical distribution:

    "Tyler 1987 - Statistical analysis for the angular central Gaussian distribution on the sphere"
    """

    def __init__(self, p: int,rank: int,init=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.p = p #dimensionality
        self.D = rank 
        if self.D == self.p:
            self.fullrank = True
        else:
            self.fullrank = False
        self.half_p = torch.tensor(p / 2)

        # log sphere surface area
        self.logSA = torch.lgamma(self.half_p) - torch.log(torch.tensor(2)) -self.half_p* torch.log(torch.tensor(np.pi))
        
        if self.fullrank is True: #cholesky
            self.L_vec = nn.Parameter(torch.randn(int(self.p*(self.p-1)/2+self.p),dtype=torch.double).to(self.device))
            
            self.tril_indices = torch.tril_indices(self.p,self.p)

            self.diag_indices = torch.zeros(self.p).type(torch.LongTensor)
            for i in range(1,self.p+1):   
                self.diag_indices[i-1] = ((i**2+i)/2)-1
        else:
            if init is None:
                self.M = nn.Parameter(torch.randn((self.p,self.D),dtype=torch.double).to(self.device))
            else:
                self.M = init
                num_missing = self.D-init.shape[1]
                M_extra = torch.randn(self.p,num_missing,dtype=torch.double)
                self.M = nn.Parameter(torch.cat([init,M_extra],dim=1))

    def get_params(self):
        if self.fullrank is True:
            L_tri_inv = torch.zeros(self.p,self.p,device=self.device,dtype=torch.double)
            L_tri_inv[self.tril_indices[0],self.tril_indices[1]] = self.L_vec
            return {'L_tri_inv':L_tri_inv}
        else:
            return {'M':self.M} #should be normalized: L=MM^T+I and then L = p*L/trace(L)

    def log_determinant_L(self,L):
        log_det_L = torch.log(torch.det(L))
        
        return log_det_L
    
    def lowrank_log_pdf(self,X):

        if self.fullrank is True:
            L_tri_inv = torch.zeros(self.p,self.p,device=self.device,dtype=torch.double)
            L_tri_inv[self.tril_indices[0],self.tril_indices[1]] = self.L_vec
            log_det_L = -2 * torch.sum(torch.log(torch.abs(self.L_vec[self.diag_indices]))) #added minus
            B = X @ L_tri_inv
            matmul2 = torch.sum(B * B, dim=1)
        else:
            L = torch.eye(self.D)+self.M.T@self.M #note DxD not pxp since invariant
            log_det_L = self.log_determinant_L(L)
            B = X@self.M
            matmul2 = 1-torch.sum(B@torch.linalg.inv(L)*B,dim=1)

        # minus log_det_L instead of + log_det_A_inv
        log_acg_pdf = self.logSA - 0.5 * log_det_L - self.half_p * torch.log(matmul2)
        return log_acg_pdf

    def forward(self, X):
        return self.lowrank_log_pdf(X)
    
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
