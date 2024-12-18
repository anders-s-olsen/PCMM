import torch
import torch.nn as nn

# import the numpy models for initializing parameters
from PCMM.PCMMnumpy import Watson, ACG, MACG, SingularWishart, Normal

class PCMMtorchBaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.LogSoftmax_pi = nn.LogSoftmax(dim=0)
        self.LogSoftmax_T = nn.LogSoftmax(dim=1)

    def unpack_params(self,params):
        if not torch.is_tensor(params[list(params.keys())[0]]):
            for key in params.keys():
                params[key] = torch.tensor(params[key])
        
        # distribution-specific settings
        if 'Watson' in self.distribution:
            self.mu = nn.Parameter(params['mu'])
            self.kappa = nn.Parameter(params['kappa'])
        elif 'lowrank' in self.distribution:
            self.M = nn.Parameter(params['M'])
        else:
            raise ValueError('Invalid distribution')
        
        if self.distribution in ['SingularWishart_lowrank','Normal_lowrank','Complex_Normal_lowrank']:
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

    def initialize_transition_matrix(self,X):
        beta = self.posterior_MM(self.log_pdf(X))
        T = torch.zeros(self.K,self.K)
        for i in range(len(X)-1):
            T += beta[:,i][:,None]*beta[:,i+1][None,:]
        T = T/T.sum(dim=1)[:,None]
        self.T = nn.Parameter(T)

    def initialize(self,X,init_method=None):
        # initialize using analytical optimization only
        if self.distribution == 'Watson':
            WatsonEM = Watson(p=self.p.item(),K=self.K.item())
            WatsonEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(WatsonEM.get_params())
        elif self.distribution == 'Complex_Watson':
            WatsonEM = Watson(p=self.p.item(),K=self.K.item(),complex=True)
            WatsonEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(WatsonEM.get_params())
        elif self.distribution == 'ACG_lowrank':
            ACGEM = ACG(p=self.p.item(),K=self.K.item(),rank=self.r.item())
            ACGEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(ACGEM.get_params())
        elif self.distribution == 'Complex_ACG_lowrank':
            ACGEM = ACG(p=self.p.item(),K=self.K.item(),rank=self.r.item(),complex=True)
            ACGEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(ACGEM.get_params())
        elif self.distribution == 'MACG_lowrank':
            MACGEM = MACG(p=self.p.item(),K=self.K.item(),rank=self.r.item(),q=self.q.item())
            MACGEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(MACGEM.get_params())
        elif self.distribution == 'SingularWishart_lowrank':
            SingularWishartEM = SingularWishart(p=self.p.item(),K=self.K.item(),q=self.q.item(),rank=self.r.item())
            SingularWishartEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(SingularWishartEM.get_params())
        elif self.distribution == 'Normal_lowrank':
            NormalEM = Normal(p=self.p.item(),K=self.K.item(),rank=self.r.item())
            NormalEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(NormalEM.get_params())
        elif self.distribution == 'Complex_Normal_lowrank':
            NormalEM = Normal(p=self.p.item(),K=self.K.item(),rank=self.r.item(),complex=True)
            NormalEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(NormalEM.get_params())
        else:
            raise ValueError('Invalid distribution')

    def MM_log_likelihood(self,log_pdf,return_samplewise_likelihood=False):
        log_density = log_pdf+self.LogSoftmax_pi(self.pi)[:,None] #each pdf gets "multiplied" with the weight
        logsum_density = torch.logsumexp(log_density,dim=0) #sum over the K components
        log_likelihood = torch.sum(logsum_density) #sum over the N samples
        if return_samplewise_likelihood:
            return log_likelihood, logsum_density
        else:
            return log_likelihood

    def HMM_log_likelihood_seq_nonuniform_sequences(self,log_pdf,return_samplewise_likelihood=False):
        K,N = log_pdf.shape
        sequence_starts = torch.hstack([torch.tensor(0),torch.cumsum(self.samples_per_sequence[:-1],0)])
        log_T = self.LogSoftmax_T(self.T) # size KxK
        log_pi = self.LogSoftmax_pi(self.pi) #size K

        log_prob = torch.zeros(len(self.samples_per_sequence))
        for seq in range(self.samples_per_sequence.size(0)):
            Ns = self.samples_per_sequence[seq]
            log_alpha = torch.zeros(Ns,K)
            log_alpha[0,:] = log_pdf[:,sequence_starts[seq]] + log_pi

            for t in range(1,Ns):
                log_alpha[t,:] = log_pdf[:,sequence_starts[seq]+t] + torch.logsumexp(log_alpha[t-1,:,None]+log_T,dim=0)
            
            log_t = torch.logsumexp(log_alpha[-1,:],dim=-1)
            log_prob[seq] = log_t
        if return_samplewise_likelihood:
            return torch.sum(log_prob), log_prob
        else:
            return torch.sum(log_prob)
    
    def HMM_log_likelihood_seq(self,log_pdf,return_samplewise_likelihood=False):
        """
        This is a faster version of HMM_log_likelihood_seq_nonuniform_sequences, but it assumes that all sequences have the same length
        """
        K,N = log_pdf.shape
        if self.samples_per_sequence == 0:
            Ns = N
        elif self.samples_per_sequence.ndim==0:
            Ns = self.samples_per_sequence
            assert N % Ns == 0, "Number of timesteps N must be divisible by sequence length Ns. Else provide a list of sequence lengths"
        elif len(self.samples_per_sequence.unique())==1:
            Ns = self.samples_per_sequence[0]
        else:
            return self.HMM_log_likelihood_seq_nonuniform_sequences(log_pdf,return_samplewise_likelihood)
        num_sequences = N//Ns
        log_pdf_reshaped = log_pdf.T.view(num_sequences,Ns,K).swapaxes(-2,-1)
        
        log_T = self.LogSoftmax_T(self.T) # size KxK
        log_pi = self.LogSoftmax_pi(self.pi) #size K
        
        log_alpha = torch.zeros(num_sequences,Ns,K)
        log_alpha[:,0,:] = log_pdf_reshaped[:,:,0] + log_pi[None]

        for t in range(1,Ns):
            # The logsumexp is over the 2nd dimension since it's over k', not k!
            log_alpha[:,t,:] = log_pdf_reshaped[:,:,t] + torch.logsumexp(log_alpha[:,t-1,:,None]+log_T[None],dim=1) 

        log_t = torch.logsumexp(log_alpha[:,-1,:],dim=-1) #this is the logsumexp over the K states
        log_prob = torch.sum(log_t,dim=0) #sum over sequences

        if return_samplewise_likelihood:
            return log_prob, log_t
        else:
            return log_prob

    def forward(self, X, return_samplewise_likelihood=False):
        log_pdf = self.log_pdf(X)
        if self.K==1:
            if return_samplewise_likelihood:
                return torch.sum(log_pdf), log_pdf[0]
            else:
                return torch.sum(log_pdf)
        else:
            if self.HMM:
                return self.HMM_log_likelihood_seq(log_pdf,return_samplewise_likelihood)
            else:
                return self.MM_log_likelihood(log_pdf,return_samplewise_likelihood)
        
    def test_log_likelihood(self,X):
        with torch.no_grad():
            return self.forward(X,return_samplewise_likelihood=True)

    def posterior_MM(self,log_pdf):
        log_density = log_pdf+self.LogSoftmax_pi(self.pi)[:,None]
        logsum_density = torch.logsumexp(log_density,dim=0)
        return torch.exp(log_density-logsum_density)

    def viterbi(self,log_pdf):
        K,N = log_pdf.shape
        if self.samples_per_sequence == 0:
            samples_per_sequence = torch.atleast_1d(torch.tensor(N))
            sequence_starts = torch.atleast_1d(torch.tensor(0))
        elif self.samples_per_sequence.ndim==0:
            samples_per_sequence = torch.atleast_1d(self.samples_per_sequence).repeat(N//samples_per_sequence)
            sequence_starts = torch.hstack([torch.tensor(0),torch.cumsum(samples_per_sequence[:-1],0)])
        else:
            samples_per_sequence = torch.tensor(self.samples_per_sequence)
        log_T = self.LogSoftmax_T(self.T) # size KxK
        log_pi = self.LogSoftmax_pi(self.pi) #size K
        
        Z_path_all = []
        for seq in range(samples_per_sequence.size(0)):
            Ns = samples_per_sequence[seq]
            log_psi = torch.zeros(Ns,K)

            log_psi[0,:] = log_pdf[:,sequence_starts[seq]]+log_pi

            for t in range(1,Ns):
                temp = log_psi[t-1,:,None]+log_T
                log_psi[t,:],_ = torch.max(temp,dim=0) #maximum over "from" states
                log_psi[t,:] = log_psi[t,:]+log_pdf[:,sequence_starts[seq]+t]

            Z_T = torch.argmax(log_psi[-1,:])
            Z_path = torch.zeros(Ns,dtype=torch.int32)
            Z_path[-1] = Z_T
            for t in range(Ns-2,-1,-1):
                Z_path[t] = torch.argmax(log_psi[t,:]+log_T[:,Z_path[t+1]])

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
                return self.viterbi(log_pdf)
            else:
                return self.posterior_MM(log_pdf)
    
    def get_params(self):
        if self.HMM:
            if 'Watson' in self.distribution:
                return {'mu':self.mu.detach(),'kappa':self.kappa.detach(),'pi':self.pi.detach(),'T':self.T.detach()}
            elif self.distribution in ['ACG_lowrank','Complex_ACG_lowrank','MACG_lowrank']:
                return {'M':self.M.detach(),'pi':self.pi.detach(),'T':self.T.detach()}
            elif self.distribution in ['SingularWishart_lowrank','Normal_lowrank','Complex_Normal_lowrank']:
                return {'M':self.M.detach(),'gamma':self.gamma.detach(),'pi':self.pi.detach(),'T':self.T.detach()}
        else:
            if 'Watson' in self.distribution:
                return {'mu':self.mu.detach(),'kappa':self.kappa.detach(),'pi':self.pi.detach()}
            elif self.distribution in ['ACG_lowrank','Complex_ACG_lowrank','MACG_lowrank']:
                return {'M':self.M.detach(),'pi':self.pi.detach()}
            elif self.distribution in ['SingularWishart_lowrank','Normal_lowrank','Complex_Normal_lowrank']:
                return {'M':self.M.detach(),'gamma':self.gamma.detach(),'pi':self.pi.detach()}
        
    def set_params(self,params):
        self.unpack_params(params)
        