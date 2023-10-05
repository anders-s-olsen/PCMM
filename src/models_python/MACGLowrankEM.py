import numpy as np
from scipy.special import loggamma, gamma
from src.models_python.diametrical_clustering import diametrical_clustering, diametrical_clustering_plusplus
from src.models_python.WatsonMixtureEM import Watson
from src.models_python.mixture_EM_loop import mixture_EM_loop
import matplotlib.pyplot as plt
import time

class MACG():
    """
    Mixture model class
    """
    def __init__(self, K: int, p: int, q: int,rank:int,params=None):
        super().__init__()
        self.K = K
        self.q = q
        self.p = p
        self.half_p = self.p/2
        self.r = rank
        loggamma_k = (self.q*(self.q-1)/4)*np.log(np.pi)+np.sum(loggamma(self.half_p-np.arange(self.q)/2))
        self.logSA_Stiefel = loggamma_k-self.q*np.log(2)-self.q*self.half_p*np.log(np.pi)

        if params is not None: # for evaluating likelihood with already-learned parameters
            self.pi = np.array(params['pi'])
            M_init = params['M']
            if M_init.dim()!=3 or M_init.shape[2]!=self.r: # add extra columns
                if M_init.dim()==2:
                    num_missing = self.r-1
                else:
                    num_missing = self.r-M_init.shape[2]

                M_extra = np.random.uniform(size=(self.K,self.p,num_missing))
                self.M = np.concatenate([M_init,M_extra],axis=2)
            else: 
                self.M = M_init

        
    def get_params(self):
        return {'M': self.M,'pi':self.pi,'c':self.c}
    
    def initialize(self,X=None,init=None,tol=None):
        self.pi = np.repeat(1/self.K,repeats=self.K)
        if init is not None and init!='uniform' and init!='unif':
            if init == '++' or init == 'plusplus' or init == 'diametrical_clustering_plusplus':
                mu = diametrical_clustering_plusplus(X=X,K=self.K)
            elif init == 'dc' or init == 'diametrical_clustering':
                mu = diametrical_clustering(X=X,K=self.K,max_iter=100000,num_repl=5,init='++',tol=tol)
            elif init == 'WMM' or init == 'Watson' or init == 'W' or init == 'watson':
                W = Watson(K=self.K,p=self.p)
                params,_,_,_ = mixture_EM_loop(W,X,init='dc')
                mu = params['mu']
                self.pi = params['pi']
            
            self.M = np.random.uniform(size=(self.K,self.p,self.r))
            for k in range(self.K):
                self.M[k,:,0] = mu[:,k] #initialize only the first of the rank D columns this way, the rest uniform
        elif init =='unif' or init=='uniform' or init is None:
            self.M = np.random.uniform(size=(self.K,self.p,self.r))
    

################ E-step ###################

    def logdet(self,B):
        logdetsign,logdet = np.linalg.slogdet(B)
        return logdetsign*logdet
    
    def log_pdf(self,X):

        D = np.eye(self.r)+np.swapaxes(self.M,-2,-1)@self.M
        log_det_D = self.logdet(D)
        XtM = np.swapaxes(X,-2,-1)[None,:,:,:]@self.M[:,None,:,:]

        # utilizing matrix determinant lemma
        v = self.logdet(D[:,None]-np.swapaxes(XtM,-2,-1)@XtM)-log_det_D

        # Z = np.array([np.eye(self.p)+self.M[k]@self.M[k].T for k in range(self.K)])
        # pdf = np.log(np.linalg.det(np.swapaxes(X,-2,-1)[None,:,:,:]@np.linalg.inv(Z)[:,None,:,:]@X))
        return self.logSA_Stiefel - (self.q/2)*self.logdet(D)[:,None] - self.half_p*v

    def log_density(self,X):
        return self.log_pdf(X)+np.log(self.pi)[:,None]
    
    def log_likelihood(self,X):
        self.density = self.log_density(X)
        self.logsum_density = np.logaddexp.reduce(self.density)
        loglik = np.sum(self.logsum_density)
        return loglik
    
    def posterior(self,X):
        density = self.log_density(X)
        logsum_density = np.logaddexp.reduce(density)
        return np.exp(density-logsum_density)    

    def M_MLE_lowrank(self,M,X,weights = None,tol=1e-10,max_iter=10000):
        n,p,q = X.shape
        
        if n<(p*(p-1)*q):
            print("Too high dimensionality compared to number of observations. Sigma cannot be calculated")
            return
        # n = 10
        # X = X[:n,:,:]
        # weights = weights[:n]
        if weights is None:
            weights = np.ones(n)
        Q = weights[:,None,None]*X

        loss = []
        o = np.linalg.norm(M,'fro')**2
        b = 1/(1+o/p)
        M = np.sqrt(b)*M
        
        trMMtMMt_old = np.trace(M.T@M@M.T@M)
        M_old = M
        o_all = [o]
        
        for j in range(max_iter):
            
            # And then matrix
            MtM = M.T@M
            D_inv = np.linalg.inv(np.eye(self.r)+MtM)
            XtM = np.swapaxes(X,-2,-1)@M

            # # this works but is slow, should be instant with einsum...
            # L,U = np.linalg.eigh(XtM@D_inv@np.swapaxes(XtM,-2,-1))
            # Ldiag = 1/L[..., np.newaxis] * np.eye(self.q)
            # V_inv = U@Ldiag@np.swapaxes(U,-2,-1)
            
            # This is okay, because the matrices to be inverted are only 2x2
            V_inv = np.linalg.inv(np.eye(q)-XtM@D_inv@np.swapaxes(XtM,-2,-1))
            
            # Works, think it's okay in terms of speed bco precomputation of XtM
            M = p/(q*np.sum(weights))*np.sum(Q@V_inv@XtM,axis=0)@(np.eye(self.r)-D_inv@MtM)
            o = np.linalg.norm(M,'fro')**2
            b = 1/(1+o/p)
            M = np.sqrt(b)*M
            o_all.append(o)

            trMMtMMt = np.trace(M.T@M@M.T@M)

            #Svarende til loss.append(np.linalg.norm(Z_old-Z)**2)
            # Kan man virkelig ikke reducere np.trace(M.T@M@M.T@M)??
            loss.append(trMMtMMt+trMMtMMt_old-2*np.trace(M@M.T@M_old@M_old.T))
            
            if j>0:
                if loss[-1]<tol:
                    break
            
            trMMtMMt_old = trMMtMMt
            M_old = M


        return M

############# M-step #################
    def M_step(self,X,tol=1e-10):
        n,p,q = X.shape
        Beta = np.exp(self.density-self.logsum_density).T
        self.pi = np.sum(Beta,axis=0)/n

        for k in range(self.K):
            self.M[k] = self.M_MLE_lowrank(self.M[k],X,weights=Beta[:,k],tol=tol)

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    K = np.array(2)
    
    p = np.array(3)
    MACG = MACG(K=K,p=3,q=2)
    data = np.loadtxt('data/synthetic/synth_data_4.csv',delimiter=',')
    data2 = np.zeros((1000,p,2))
    data2[:,:,0] = data[np.arange(2000,step=2),:]
    data2[:,:,1] = data[np.arange(2000,step=2)+1,:]
    # data = np.random.normal(loc=0,scale=0.1,size=(10000,100))
    # data = data[np.arange(2000,step=2),:]
    MACG.initialize(X=data2,init='uniform')


    for iter in tqdm(range(1000)):
        # E-step
        MACG.log_likelihood(X=data2)
        # print(ACG.Lambda_chol)
        # M-step
        MACG.M_step(X=data2)
    stop=7


# old way, Z becomes too big
# c = self.c[0]
# Z = c*np.eye(self.p)+self.M[0]@self.M[0].T
# start_time = time.time()
# a1 = np.linalg.det(np.swapaxes(X,-2,-1)@np.linalg.inv(Z)@X)
# end_time = time.time()
# elapsed1 = end_time-start_time

# # new, basic way, equal to old way
# start_time = time.time()
# a2 = np.linalg.det(c**(-1)*np.eye(self.q)-c**(-2)*np.swapaxes(X,-2,-1)@self.M[0]@np.linalg.inv(D[0])@self.M[0].T@X)
# end_time = time.time()
# elapsed2 = end_time-start_time

# # new, better way using matrix determinant lemma, avoiding inverting D
# start_time = time.time()
# a3=np.linalg.det(D[0]-c*self.M[0].T@X@np.swapaxes(X,-2,-1)@self.M[0])/np.linalg.det(D[0])*np.linalg.det(c**(-1)*np.eye(self.q))
# end_time = time.time()
# elapsed3 = end_time-start_time

# # Slightly shortened with precomputed XM
# start_time = time.time()
# a4=np.linalg.det(D[0]-c*np.swapaxes(XM[0],-2,-1)@XM[0])/np.linalg.det(D[0])*np.linalg.det(c**(-1)*np.eye(self.q))
# end_time = time.time()
# elapsed4 = end_time-start_time