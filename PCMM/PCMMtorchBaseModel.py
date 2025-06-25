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
                params[key] = torch.as_tensor(params[key])
        
        # distribution-specific settings
        if 'Watson' in self.distribution:
            self.mu = nn.Parameter(params['mu'])
            self.kappa = nn.Parameter(params['kappa'])
        elif 'lowrank' in self.distribution:
            self.M = nn.Parameter(params['M'])
            if self.M.shape[0]!=self.K:
                raise ValueError('Number of components in params[''M''] doesn''t match the number of components specified')
            if self.M.shape[1]!=self.p:
                raise ValueError('The dimensionality of M doesn''t match the specified dimensionality p')
            if self.M.shape[2]>self.r:
                raise ValueError('M should be Kxpxr, where r is the rank of the model. ')
            if torch.is_complex(self.M) and not 'Complex' in self.distribution:
                raise ValueError('Model specified to not be complex-valued but params[''M''] is complex-valued')
            if not torch.is_complex(self.M) and 'Complex' in self.distribution:
                raise ValueError('Model specified to be complex-valued but params[''M''] is real-valued')
        else:
            raise ValueError('Invalid distribution')
        
        if self.distribution in ['SingularWishart_lowrank','Normal_lowrank','Complex_Normal_lowrank']:
            if 'gamma' in params:
                self.gamma = nn.Parameter(params['gamma'])
            else:
                self.gamma = nn.Parameter(torch.ones(self.K))

        # mixture or HMM settings
        self.pi = nn.Parameter(params['pi'])

        if self.HMM:
            if 'T' in params:
                self.T = nn.Parameter(params['T'])
                # otherwise T will be initialized in mixture_torch_loop

    def _format_samples_per_sequence(self,N):
        if torch.all(self.samples_per_sequence == 0):
            samples_per_sequence = torch.atleast_1d(torch.tensor(N))
            sequence_starts = torch.atleast_1d(torch.tensor(0))
        elif self.samples_per_sequence.ndim==0:
            samples_per_sequence = torch.atleast_1d(self.samples_per_sequence).repeat(N//self.samples_per_sequence)
            sequence_starts = torch.hstack([torch.tensor(0),torch.cumsum(samples_per_sequence[:-1],0)])
        elif torch.remainder(N,self.samples_per_sequence.sum())==0:
            samples_per_sequence = self.samples_per_sequence.repeat(N//self.samples_per_sequence.sum())
            sequence_starts = torch.hstack([torch.tensor(0),torch.cumsum(samples_per_sequence[:-1],0)])
        else:
            samples_per_sequence = self.samples_per_sequence
            sequence_starts = torch.hstack([torch.tensor(0),torch.cumsum(samples_per_sequence[:-1],0)])
        
        if torch.sum(samples_per_sequence) != N:
            raise ValueError('Number of samples in samples_per_sequence does not match the number of samples in X. Please check your input data.')

        return samples_per_sequence, sequence_starts

    def initialize_transition_matrix_hmm(self,X):
        with torch.no_grad():
            beta = self.posterior_MM(self.log_pdf(X,recompute_statics=True)) # KxN
            K,N = beta.shape
            samples_per_sequence, sequence_starts = self._format_samples_per_sequence(N)
            
            T = torch.zeros(samples_per_sequence.size(0),self.K,self.K)
            delta = torch.zeros(samples_per_sequence.size(0),self.K)
            for seq in range(samples_per_sequence.size(0)):
                for t in range(samples_per_sequence[seq]-1):
                    T[seq] += 1/samples_per_sequence[seq] * torch.outer(beta[:,sequence_starts[seq]+t],beta[:,sequence_starts[seq]+t+1])
                delta[seq] = beta[:,sequence_starts[seq]]
            delta = delta.mean(dim=0)
            delta = delta / delta.sum() # normalize to sum to 1
            T = torch.mean(T,dim=0) # average over sequences
            # T = torch.zeros(self.K,self.K)
            # for i in range(len(X)-1):
            #     T += beta[:,i][:,None]*beta[:,i+1][None,:]
            T = T/T.sum(dim=1)[:,None]
            # self.T = nn.Parameter(T)
            return T, delta

    def initialize(self,X,init_method=None):
        # initialize using analytical optimization only
        if self.distribution == 'Watson':
            WatsonEM = Watson(p=self.p,K=self.K)
            WatsonEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(WatsonEM.get_params())
        elif self.distribution == 'Complex_Watson':
            WatsonEM = Watson(p=self.p,K=self.K,complex=True)
            WatsonEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(WatsonEM.get_params())
        elif self.distribution == 'ACG_lowrank':
            ACGEM = ACG(p=self.p,K=self.K,rank=self.r)
            ACGEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(ACGEM.get_params())
        elif self.distribution == 'Complex_ACG_lowrank':
            ACGEM = ACG(p=self.p,K=self.K,rank=self.r,complex=True)
            ACGEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(ACGEM.get_params())
        elif self.distribution == 'MACG_lowrank':
            MACGEM = MACG(p=self.p,K=self.K,rank=self.r,q=self.q)
            MACGEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(MACGEM.get_params())
        elif self.distribution == 'SingularWishart_lowrank':
            SingularWishartEM = SingularWishart(p=self.p,K=self.K,q=self.q,rank=self.r)
            SingularWishartEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(SingularWishartEM.get_params())
        elif self.distribution == 'Normal_lowrank':
            NormalEM = Normal(p=self.p,K=self.K,rank=self.r)
            NormalEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(NormalEM.get_params())
        elif self.distribution == 'Complex_Normal_lowrank':
            NormalEM = Normal(p=self.p,K=self.K,rank=self.r,complex=True)
            NormalEM.initialize(X.numpy(),init_method=init_method)
            self.unpack_params(NormalEM.get_params())
        else:
            raise ValueError('Invalid distribution')

    def MM_log_likelihood(self,log_pdf,return_samplewise_likelihood=False):
        log_density = log_pdf+self.LogSoftmax_pi(self.pi)[:,None] #each pdf gets "multiplied" with the weight
        logsum_density = torch.logsumexp(log_density,dim=0) #sum over the K components
        log_likelihood = torch.sum(logsum_density) #sum over the N samples
        if return_samplewise_likelihood:
            return log_likelihood, logsum_density.numpy()
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

            log_alpha = log_pdf[:,sequence_starts[seq]] + log_pi #add the log_pdf to the log_alpha
            for t in range(1,Ns):
                log_alpha = log_pdf[:,sequence_starts[seq]+t] + torch.logsumexp(log_alpha[:,None]+log_T,dim=0)

            log_prob[seq] = torch.logsumexp(log_alpha,-1)
        if return_samplewise_likelihood:
            return torch.sum(log_prob), torch.nan
        else:
            return torch.sum(log_prob)

    def HMM_log_likelihood_seq_nonuniform_sequences_batch(self,log_pdf,return_samplewise_likelihood=False):
        K,N = log_pdf.shape
        num_subs = N // self.samples_per_sequence.sum()
        log_pdf_reshaped = log_pdf.T.view(num_subs,self.samples_per_sequence.sum(),self.K).swapaxes(-2,-1) # reshape to BxKxN
        
        sequence_starts = torch.hstack([torch.tensor(0),torch.cumsum(self.samples_per_sequence[:-1],0)])
        log_T = self.LogSoftmax_T(self.T) # size KxK
        log_pi = self.LogSoftmax_pi(self.pi) #size K

        log_prob = torch.zeros(len(self.samples_per_sequence))
        log_t_sub = torch.zeros(num_subs) # to store the log likelihood for each subject
        for seq in range(self.samples_per_sequence.size(0)):
            Ns = self.samples_per_sequence[seq]

            log_alpha = log_pdf_reshaped[:,:,0] + log_pi[None] #add the log_pdf to the log_alpha
            for t in range(1,Ns):
                log_alpha = log_pdf_reshaped[:,:,sequence_starts[seq]+t] + torch.logsumexp(log_alpha[:,:,None]+log_T[None],dim=1)

            log_t = torch.logsumexp(log_alpha,-1) #this is the logsumexp over the K states
            log_t_sub += log_t # accumulate the log likelihood for each subject
            log_prob[seq] = torch.sum(log_t) #sum over subjects
        
        if return_samplewise_likelihood:
            return torch.sum(log_prob), log_t_sub.numpy()
        else:
            return torch.sum(log_prob)
    
    def HMM_log_likelihood_seq(self,log_pdf,return_samplewise_likelihood=False):
        """
        This is a faster version of HMM_log_likelihood_seq_nonuniform_sequences, but it assumes that all sequences have the same length
        """
        K,N = log_pdf.shape
        if torch.all(self.samples_per_sequence == 0):
            Ns = N
        elif self.samples_per_sequence.ndim==0:
            Ns = self.samples_per_sequence
            assert N % Ns == 0, "Number of timesteps N must be divisible by sequence length Ns. Else provide a list of sequence lengths"
        elif len(self.samples_per_sequence.unique())==1:
            Ns = self.samples_per_sequence[0]
        elif torch.remainder(N,self.samples_per_sequence.sum())==0:
            return self.HMM_log_likelihood_seq_nonuniform_sequences_batch(log_pdf,return_samplewise_likelihood)
        else:
            return self.HMM_log_likelihood_seq_nonuniform_sequences(log_pdf,return_samplewise_likelihood)
        num_sequences = N//Ns
        log_pdf_reshaped = log_pdf.T.view(num_sequences,Ns,K).swapaxes(-2,-1)
        
        log_T = self.LogSoftmax_T(self.T) # size KxK
        log_pi = self.LogSoftmax_pi(self.pi) #size K
        
        log_alpha = log_pdf_reshaped[:,:,0] + log_pi[None] #add the log_pdf to the log_alpha
        for t in range(1,Ns):
            log_alpha = log_pdf_reshaped[:,:,t] + torch.logsumexp(log_alpha[:,:,None]+log_T[None],dim=1)

        log_t = torch.logsumexp(log_alpha,-1) #this is the logsumexp over the K states
        log_prob = torch.sum(log_t) #sum over sequences

        if return_samplewise_likelihood:
            return log_prob, log_t.numpy()
        else:
            return log_prob

    def forward(self, X, return_samplewise_likelihood=False, recompute_statics=False):
        log_pdf = self.log_pdf(X,recompute_statics=recompute_statics)
        if self.K==1:
            if return_samplewise_likelihood:
                return torch.sum(log_pdf), log_pdf[0].numpy()
            else:
                return torch.sum(log_pdf)
        else:
            if self.HMM:
                return self.HMM_log_likelihood_seq(log_pdf,return_samplewise_likelihood)
            else:
                return self.MM_log_likelihood(log_pdf,return_samplewise_likelihood)
        
    def test_log_likelihood(self,X):
        with torch.no_grad():
            return self.forward(X,return_samplewise_likelihood=True,recompute_statics=True)

    def posterior_MM(self,log_pdf):
        log_density = log_pdf+self.LogSoftmax_pi(self.pi)[:,None]
        logsum_density = torch.logsumexp(log_density,dim=0)
        return torch.exp(log_density-logsum_density)

    def viterbi(self,log_pdf):
        K,N = log_pdf.shape
        samples_per_sequence, sequence_starts = self._format_samples_per_sequence(N)

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
            Z_path_all.append(torch.eye(K)[Z_path].T)

        return torch.hstack(Z_path_all)
    
    def posterior(self,X):
        with torch.no_grad():
            log_pdf = self.log_pdf(X)
            if self.HMM:
                return self.viterbi(log_pdf).numpy()
            else:
                return self.posterior_MM(log_pdf).numpy()
    
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
        