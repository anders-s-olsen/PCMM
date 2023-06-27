import numpy as np
import torch
import torch.nn as nn

#from scipy.special import gamma

#device = 'cpu'
class AngularCentralGaussian(nn.Module):
    """
    Angular-Central-Gaussian spherical distribution:

    "Tyler 1987 - Statistical analysis for the angular central Gaussian distribution on the sphere"
    """

    def __init__(self, p,D,init=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.p = p
        self.D = D
        self.half_p = torch.tensor(p / 2)

        # log sphere surface area
        self.logSA = torch.lgamma(self.half_p) - torch.log(torch.tensor(2)) -self.half_p* torch.log(torch.tensor(np.pi))
        if init is None:
            self.L = nn.Parameter(torch.randn(self.p,self.D,dtype=torch.double).to(self.device))
        else:
            self.L = init
            num_missing = self.D-init.shape[1]
            L_extra = torch.randn(self.p,num_missing,dtype=torch.double)
            self.L = nn.Parameter(torch.cat([init,L_extra],dim=1))
        assert self.p != 1, 'Not matmul not stable for this dimension'

    def get_params(self):
        return self.lowrank_compose_A(read_A_param=True)

    def lowrank_compose_A(self,read_A_param=False):
        
        if read_A_param:
            return self.L

        log_det_A = 2 * torch.sum(torch.log(torch.abs(torch.diag(torch.linalg.cholesky(torch.eye(self.D)+self.L.T@self.L)))))
        
        return log_det_A
    
    def lowrank_log_pdf(self,X):
        log_det_A = self.lowrank_compose_A()

        B = X@self.L
        matmul2 = 1-torch.sum(B@torch.linalg.inv(torch.eye(self.D)+self.L.T@self.L)*B,dim=1)

        # minus log_det_A instead of + log_det_A_inv
        log_acg_pdf = self.logSA - 0.5 * log_det_A - self.half_p * torch.log(matmul2)
        return log_acg_pdf

    def forward(self, X):
        return self.lowrank_log_pdf(X)
    
    def __repr__(self):
        return 'ACG'


if __name__ == "__main__":
    # test that the code works
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scipy

    mpl.use('Qt5Agg')
    dim = 3
    ACG = AngularCentralGaussian(p=dim)
    
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
