import torch
import torch.nn as nn
from src.DMM_EM.WatsonEM import Watson
from src.DMM_EM.ACGEM import ACG
from src.DMM_EM.MACGEM import MACG
from src.DMM_EM.SingularWishartEM import SingularWishart

class DMMPyTorchBaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.LogSoftmax_pi = nn.LogSoftmax(dim=0)
        self.LogSoftmax_T = nn.LogSoftmax(dim=1)

    def unpack_params(self,params):
        if not torch.is_tensor(params[list(params.keys())[0]]):
            for key in params.keys():
                params[key] = torch.tensor(params[key])
        
        # distribution-specific settings
        if self.distribution in ['Watson','Complex_Watson']:
            self.mu = nn.Parameter(params['mu'])
            self.kappa = nn.Parameter(params['kappa'])
        elif self.distribution in ['ACG_lowrank','Complex_ACG_lowrank','MACG_lowrank','SingularWishart_lowrank']:
            self.M = params['M']
        else:
            raise ValueError('Invalid distribution')
        
        if self.distribution == 'SingularWishart_lowrank':
            if 'gamma' in params:
                self.gamma = nn.Parameter(params['gamma'])
            else:
                self.gamma = nn.Parameter(torch.ones(self.K))

        # mixture or HMM settings
        if 'pi' in params:
            self.pi = nn.Parameter(params['pi'])
        else:
            self.pi = nn.Parameter(1/self.K.repeat(self.K))

        if self.HMM:
            if 'T' in params:
                self.T = nn.Parameter(params['T'])
            else:
                self.T = nn.Parameter(1/self.K.repeat(self.K,self.K))

    def compute_transition_matrix(self,seq):
        T = torch.zeros(self.K,self.K)
        for i in range(len(seq)-1):
            T[seq[i],seq[i+1]] += 1
        T = T/T.sum(dim=1)[:,None]
        return T

    def initialize_transition_matrix(self,X):
        X_part = torch.argmax(self.posterior_MM(self.log_pdf(X)),dim=0)
        for k in range(self.K):
            if torch.sum(X_part==k)==0:
                self.T = nn.Parameter(1/self.K.repeat(self.K,self.K))
                break
        self.T = nn.Parameter(self.compute_transition_matrix(X_part))

    def initialize(self,X,init_method=None,L=None):
        # initialize using analytical optimization only
        if self.distribution == 'Watson':
            WatsonEM = Watson(p=self.p.numpy(),K=self.K.numpy())
            WatsonEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(WatsonEM.get_params())
        elif self.distribution == 'Complex_Watson':
            WatsonEM = Watson(p=self.p.numpy(),K=self.K.numpy(),complex=True)
            WatsonEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(WatsonEM.get_params())
        elif self.distribution == 'ACG_lowrank':
            ACGEM = ACG(p=self.p.numpy(),K=self.K.numpy(),rank=self.r.numpy())
            ACGEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(ACGEM.get_params())
        elif self.distribution == 'Complex_ACG_lowrank':
            ACGEM = ACG(p=self.p.numpy(),K=self.K.numpy(),rank=self.r.numpy(),complex=True)
            ACGEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(ACGEM.get_params())
        elif self.distribution == 'MACG_lowrank':
            MACGEM = MACG(p=self.p.numpy(),K=self.K.numpy(),rank=self.r.numpy(),q=self.q.numpy())
            MACGEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(MACGEM.get_params())
        elif self.distribution == 'SingularWishart_lowrank':
            SingularWishartEM = SingularWishart(p=self.p.numpy(),K=self.K.numpy(),q=self.q.numpy(),rank=self.r.numpy())
            SingularWishartEM.initialize(X.numpy(),init_method=init_method,L=L.numpy())
            self.unpack_params(SingularWishartEM.get_params())

    def MM_log_likelihood(self,log_pdf):
        log_density = log_pdf+self.LogSoftmax_pi(self.pi)[:,None] #each pdf gets "multiplied" with the weight
        logsum_density = torch.logsumexp(log_density,dim=0) #sum over the K components
        log_likelihood = torch.sum(logsum_density) #sum over the N samples
        return log_likelihood

    def HMM_log_likelihood_seq_nonuniform_sequences(self,log_pdf,samples_per_sequence):
        K,N = log_pdf.shape
        sequence_starts = torch.hstack([torch.tensor(0),torch.cumsum(torch.tensor(samples_per_sequence)[:-1],0)])
        log_T = self.LogSoftmax_T(self.T) # size KxK
        log_pi = self.LogSoftmax_pi(self.pi) #size K

        log_prob = torch.zeros(len(samples_per_sequence))
        for seq in range(len(samples_per_sequence)):
            Ns = samples_per_sequence[seq]
            log_alpha = torch.zeros(Ns,K)
            log_alpha[0,:] = log_pdf[:,sequence_starts[seq]] + log_pi

            for t in range(1,Ns):
                log_alpha[t,:] = log_pdf[:,sequence_starts[seq]+t] + torch.logsumexp(log_alpha[t-1,:,None]+log_T,dim=0)
            
            log_t = torch.logsumexp(log_alpha[-1,:],dim=-1)
            log_prob[seq] = log_t
        return torch.sum(log_prob)
    
    def HMM_log_likelihood_seq(self,log_pdf,samples_per_sequence):
        """
        This is a faster version of HMM_log_likelihood_seq_nonuniform_sequences, but it assumes that all sequences have the same length
        """
        K,N = log_pdf.shape
        if samples_per_sequence == 0:
            Ns = N
        elif torch.tensor(samples_per_sequence).ndim==0:
            Ns = samples_per_sequence
        elif len(samples_per_sequence.unique())==1:
            Ns = samples_per_sequence[0]
        else:
            return self.HMM_log_likelihood_seq_nonuniform_sequences(log_pdf,samples_per_sequence)
        log_pdf_reshaped = log_pdf.T.view(N//Ns,Ns,K).swapaxes(-2,-1)
        
        log_T = self.LogSoftmax_T(self.T) # size KxK
        log_pi = self.LogSoftmax_pi(self.pi) #size K
        
        log_alpha = torch.zeros(log_pdf_reshaped.shape[0],Ns,K)
        log_alpha[:,0,:] = log_pdf_reshaped[:,:,0] + log_pi[None]

        for t in range(1,Ns):
            # The logsumexp is over the 2nd dimension since it's over k', not k!
            log_alpha[:,t,:] = log_pdf_reshaped[:,:,t] + torch.logsumexp(log_alpha[:,t-1,:,None]+log_T[None],dim=1) 

        log_t = torch.logsumexp(log_alpha[:,-1,:],dim=-1) #this is the logsumexp over the K states
        log_prob = torch.sum(log_t,dim=0) #sum over sequences
        return log_prob

    def forward(self, X):
        log_pdf = self.log_pdf(X)
        if self.K==1:
            return torch.sum(log_pdf)
        else:
            if self.HMM:
                return self.HMM_log_likelihood_seq(log_pdf,self.samples_per_sequence)
            else:
                return self.MM_log_likelihood(log_pdf)
        
    def test_log_likelihood(self,X):
        with torch.no_grad():
            return self.forward(X)
        
    def posterior_MM(self,log_pdf):
        log_density = log_pdf+self.LogSoftmax_pi(self.pi)[:,None]
        logsum_density = torch.logsumexp(log_density,dim=0)
        return torch.exp(log_density-logsum_density)

    def viterbi2(self,log_pdf,samples_per_sequence):
        K,N = log_pdf.shape
        if samples_per_sequence == 0:
            samples_per_sequence = [torch.tensor(N)]
            sequence_starts = torch.atleast_1d(torch.tensor(0))
        elif samples_per_sequence.ndim==0:
            samples_per_sequence = samples_per_sequence.repeat(N//samples_per_sequence)
            sequence_starts = torch.hstack([torch.tensor(0),torch.cumsum(samples_per_sequence[:-1],0)])
        log_T = self.LogSoftmax_T(self.T) # size KxK
        log_pi = self.LogSoftmax_pi(self.pi) #size K
        
        Z_path_all = []
        for seq in range(len(samples_per_sequence)):
            Ns = samples_per_sequence[seq]
            log_delta = torch.zeros(Ns,K)
            psi = torch.zeros(Ns,K,dtype=torch.int32) #state sequence = integers

            log_delta[0,:] = log_pdf[:,sequence_starts[seq]]+log_pi

            for t in range(1,Ns):
                temp = log_delta[t-1,:,None]+log_T
                log_delta[t,:],psi[t,:] = torch.max(temp,dim=0) #maximum over "from" states
                log_delta[t,:] = log_delta[t,:]+log_pdf[:,sequence_starts[seq]+t]

            P_T, Z_T = torch.max(log_delta[-1,:],dim=0)
            Z_path = torch.zeros(Ns,dtype=torch.int32)
            Z_path[-1] = Z_T
            for t in range(Ns-2,-1,-1):
                Z_path[t] = psi[t+1,Z_T]

            # from partition vector to partition matrix
            Z_path2 = torch.zeros(K,Ns,dtype=torch.bool)
            for t in range(Ns):
                Z_path2[Z_path[t],t] = True
            Z_path_all.append(Z_path2)

        return torch.hstack(Z_path_all)
    
    def posterior(self,X):
        with torch.no_grad():
            log_pdf = self.log_pdf(X)
            if self.HMM:
                return self.viterbi2(log_pdf,self.samples_per_sequence)
            else:
                return self.posterior_MM(log_pdf)
    
    def get_params(self):
        if self.distribution in ['Watson','Complex_Watson']:
            return {'mu':self.mu.detach(),'kappa':self.kappa.detach(),'pi':self.pi.detach()}
        elif self.distribution in ['ACG_lowrank','Complex_ACG_lowrank','MACG_lowrank']:
            return {'M':self.M.detach(),'pi':self.pi.detach()}
        elif self.distribution == 'SingularWishart_lowrank':
            return {'M':self.M.detach(),'gamma':self.gamma.detach(),'pi':self.pi.detach()}
        
    def set_params(self,params):
        self.unpack_params(params)
        