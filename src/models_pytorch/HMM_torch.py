import torch
import torch.nn as nn
import numpy as np

class HiddenMarkovModel(nn.Module):
    """
    Hidden Markov model w. Continuous observation density
    """

    def __init__(self, K, emission_dist, observation_dim: int = 90,D=None,init=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.K = K
        self.D = D
        self.emission_density = emission_dist
        self.obs_dim = observation_dim

        if init is None:
            self.state_priors = nn.Parameter(torch.rand(self.K,device=self.device))
            self.transition_matrix = nn.Parameter(torch.rand(self.K, self.K,device=self.device))
            self.emission_models = nn.ModuleList([self.emission_density(self.obs_dim,self.D) for _ in range(self.K)])
        else:
            self.state_priors = nn.Parameter(init['pi'])
            self.transition_matrix = nn.Parameter(init['T'])
            self.emission_models = nn.ModuleList([self.emission_density(self.obs_dim,self.D,init['comp'][k]) for k in range(self.K)])

        self.softplus = nn.Softplus()
        self.logsoftmax_transition = nn.LogSoftmax(dim=1)
        self.logsoftmax_prior = nn.LogSoftmax(dim=0)

    @torch.no_grad()
    def get_model_param(self):
        priors_softmax = self.state_priors.data

        mixture_param_dict = {'un_norm_priors': priors_softmax}
        mixture_param_dict['un_norm_Transition_matrix'] = self.transition_matrix.data
        for comp_id, comp_param in enumerate(self.emission_models):
            mixture_param_dict[f'emission_model_{comp_id}'] = comp_param.get_params()
        return mixture_param_dict

    def emission_models_forward(self, X):
        return torch.stack([state_emission(X) for state_emission in self.emission_models])

    def forward(self, X):
        """
                Forward algorithm for a HMM - Solving 'Problem 1'

        :param X: (num_subject/batch_size, observation_sequence, sample_x(dim=obs_dim))
        :return: log_prob
        """
        # see (Rabiner, 1989)
        # init  1)

        log_A = self.logsoftmax_transition(self.transition_matrix)
        log_pi = self.logsoftmax_prior(self.state_priors)
        num_subjects = X.shape[0]
        seq_max = X.shape[1]
        log_alpha = torch.zeros(num_subjects, seq_max, self.K,device=self.device)

        emissions = torch.reshape(self.emission_models_forward(torch.concatenate([X[:,t,:] for t in range(seq_max)])),(self.K,seq_max,num_subjects))
        
        # time t=0
        # log_pi: (n states priors)
        # emission forward return: -> transpose -> (subject, [state1_prop(x)...stateN_prop(x)])
        log_alpha[:, 0, :] = log_pi + emissions[:,0,:].T

        # Induction 2)
        # for time:  t = 1 -> seq_max

        for t in range(1, seq_max):
            log_alpha[:, t, :] = emissions[:, t, :].T \
                                 + torch.logsumexp(log_alpha[:, t - 1, :, None] + log_A, dim=1)

        # Termination 3)
        # LogSum for states N for each time t.
        log_t_sums = torch.logsumexp(log_alpha, dim=2)

        # Retrive the alpha for the last time t in the seq, per subject
        log_props = torch.gather(log_t_sums, dim=1,
                                 index=torch.tensor([[seq_max - 1]] * num_subjects,device=self.device)).squeeze()
        # faster on GPU than just indexing...according to stackoverflow

        return log_props.sum(dim=0)  # return sum of log_prop for all subjects

    def viterbi2(self, X, external_eval=False):
        # init 1)
        log_A = self.logsoftmax_transition(self.transition_matrix)
        log_pi = self.logsoftmax_prior(self.state_priors)  # log_pi: (n states priors)

        if external_eval:
            print('Using external param setting')
            log_A = torch.log(self.transition_matrix)
            log_pi = torch.log(self.state_priors)

        num_subjects = X.shape[0]
        seq_len = X.shape[1]

        subject_Z_path = []
        subject_Z_path_prob = []
        subject_emissions = torch.zeros(num_subjects, seq_len, self.K,device=self.device)

        for subject in range(num_subjects):
            with torch.no_grad():
                X_sub = X[subject]

                # collectors
                log_delta = torch.zeros(seq_len, self.K,device=self.device)
                psi = torch.zeros(seq_len, self.K, dtype=torch.int32,device=self.device)  # intergers - state seqeunces

                # Init - time t=0
                subject_emissions[subject,0,:] = self.emission_models_forward(X_sub[0].unsqueeze(dim=0)).squeeze()
                #states_emission_0 = self.emission_models_forward(X_sub[0].unsqueeze(dim=0)).squeeze()
                log_delta[0, :] = log_pi + subject_emissions[subject,0,:]

                # Recursion 2)
                # for time:  t = 1 -> seq_max
                for t in range(1, seq_len):
                    subject_emissions[subject,t,:] = self.emission_models_forward(X_sub[t].unsqueeze(dim=0)).squeeze()
                    expression = (log_delta[t - 1, :, None] + log_A) + subject_emissions[subject,t,:]
                    
                    max_value, arg_max = torch.max(expression, dim=1)

                    log_delta[t, :] = max_value
                    psi[t, :] = arg_max

                # Termination 3) & Path backtracking 4)
                # max value and argmax at each time t, per subject.

                T_log_delta = log_delta[-1]

                Z_T_prob, Z_T = torch.max(T_log_delta, dim=0)

                Z_path = [Z_T.item()]

                for t in range(seq_len - 2, -1, -1):
                    state_t = psi[t + 1, Z_path[0]].item()

                    Z_path.insert(0, state_t)

                subject_Z_path.append(Z_path)
                subject_Z_path_prob.append(Z_T_prob)

        return np.array(subject_Z_path), np.array(subject_Z_path_prob),np.array(subject_emissions)


# if __name__ == '__main__':
#     # Test that the code works
    # from Watson_torch import Watson
    # from AngularCentralGauss_chol import AngularCentralGaussian as ACG

    # torch.manual_seed(5)
    # dim = 3

    # HMM = HiddenMarkovModel(num_states=3, observation_dim=dim, emission_dist=ACG)
    # X = torch.randint(1, 8, (2, 8, dim), dtype=torch.float)  # num_subject, seq_max, observation_dim

    # out = HMM(X)
    # #seq, probs = HMM.viterbi2(X)
    # #print(X)
    # #print(seq)
    # #print(probs)
