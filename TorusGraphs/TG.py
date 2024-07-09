import torch
import torch.nn as nn

class TorusGraphs(nn.Module):
    def __init__(self, p:int, K:int):
        super().__init__()


        self.K = K
        self.p = p
        z = p*(p-1)//2
        self.K = 1
        self.theta = nn.Parameter(torch.randn(self.K,2,z))
        self.logc = nn.Parameter(torch.zeros(self.K))


        self.triu_indices = torch.triu_indices(p,p,offset=1)


    def NCE_objective_function(self,X,noise):
        # theta (Mx2xz)
        # x (Nxp)
        N = X.shape[0]
        M = noise.shape[0] #number of noise samples


        log_prob_data = torch.zeros(self.K,N)
        log_prob_noise = torch.zeros(self.K,N)
        for k in range(self.K):
            for i in range(N):
                for z in range(self.p*(self.p-1)//2):
                    cosx = torch.cos(X[i,self.triu_indices[0,z]] - X[i,self.triu_indices[1,z]])
                    sinx = torch.sin(X[i,self.triu_indices[0,z]] - X[i,self.triu_indices[1,z]])
                    log_prob_data[k,i] += self.theta[k,:,z]@torch.tensor([cosx,sinx])


                    cosn = torch.cos(noise[i,self.triu_indices[0,z]] - noise[i,self.triu_indices[1,z]])
                    sinn = torch.sin(noise[i,self.triu_indices[0,z]] - noise[i,self.triu_indices[1,z]])
                    log_prob_noise[k,i] += self.theta[k,:,z]@torch.tensor([cosn,sinn])
        
        log_prob_data = torch.logsumexp(log_prob_data,dim=0)
        log_prob_noise = torch.logsumexp(log_prob_noise,dim=0)


        log_nx = torch.rand(N) #needs to be implemented
        log_ny = torch.rand(N)


        log_J1_denom = torch.logsumexp(torch.stack([torch.log(torch.tensor(N))+log_prob_data,torch.log(torch.tensor(M))+log_nx],dim=-1), dim=-1)
        log_J2_denom = torch.logsumexp(torch.stack([torch.log(torch.tensor(N))+log_prob_noise,torch.log(torch.tensor(M))+log_ny],dim=-1), dim=-1)


        J = torch.mean(torch.log(torch.tensor(N))+log_prob_data+self.logc-log_J1_denom,dim=-1)+torch.mean(torch.log(torch.tensor(N))+log_ny+self.logc-log_J2_denom,dim=-1)


        return J