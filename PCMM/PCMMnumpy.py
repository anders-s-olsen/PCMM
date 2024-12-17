import numpy as np
from PCMM.PCMMnumpyBaseModel import PCMMnumpyBaseModel
from scipy.sparse.linalg import svds
from scipy.optimize import minimize_scalar
from scipy.special import loggamma
from scipy.linalg import sqrtm

class Watson(PCMMnumpyBaseModel):
    def __init__(self, p:int, K:int=1, complex:bool=False, params:dict=None):
        super().__init__()

        self.K = K
        self.p = p
        
        if complex:
            self.distribution = 'Complex_Watson'
            self.a = 1
            self.c = self.p
        else:
            self.distribution = 'Watson'
            self.a = 0.5
            self.c = self.p/2

        self.logSA_sphere = loggamma(self.c) - np.log(2) - self.c* np.log(np.pi)

        self.flag_normalized_input_data = False #flag to check if the input data is normalized

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)

    def kummer_log(self,k,a,c):
        n = 1e7
        tol = 1e-10
        logkum = 0
        logkum_old = 1
        foo = 0
        if k<0:
            a = c-a
        j = 1
        while np.abs(logkum - logkum_old) > tol and (j < n):
            logkum_old = logkum
            foo += np.log((a + j - 1) / (j * (c + j - 1)) * np.abs(k))
            logkum = np.logaddexp(logkum,foo)
            j += 1
        return logkum
    
    def log_norm_constant(self):
        logkum = np.zeros(self.K)
        for idx,kappa in enumerate(self.kappa):
            logkum[idx] = self.kummer_log(k=kappa,a=self.a,c=self.c)
        return self.logSA_sphere - logkum
    
    def log_pdf(self, X):
        if not self.flag_normalized_input_data:
            if not np.allclose(np.linalg.norm(X,axis=1),1):
                raise ValueError("For the Watson distribution, the input data vectors should be normalized to unit length.")
            else:
                self.flag_normalized_input_data = True
        #the abs added for support of complex arrays
        log_pdf = self.log_norm_constant()[:,None] + self.kappa[:,None]*(np.abs(X@self.mu.conj().T)**2).T
        return log_pdf
    
    def M_step_single_component(self,X,beta,mu,kappa,tol=1e-10):
        n,p = X.shape

        if n>p:
            if kappa>0:
                _,_,mu = svds(np.sqrt(beta)[:,None]*X,k=1,which='LM',v0=mu,return_singular_vectors='vh')
            elif kappa<0:
                _,_,mu = svds(np.sqrt(beta)[:,None]*X,k=1,which='SM',v0=mu,return_singular_vectors='vh')
        else: 
            if kappa>0:
                _,_,mu = svds(np.sqrt(beta)[:,None]*X,k=1,which='LM',return_singular_vectors='vh')
            elif kappa<0:
                _,_,mu = svds(np.sqrt(beta)[:,None]*X,k=1,which='SM',return_singular_vectors='vh')

        rk = 1/np.sum(beta)*np.sum(np.abs(mu.conj()@(np.sqrt(beta)[:,None]*X).T)**2)
        LB = (rk*self.c-self.a)/(rk*(1-rk))*(1+(1-rk)/(self.c-self.a))
        B  = (rk*self.c-self.a)/(2*rk*(1-rk))*(1+np.sqrt(1+4*(self.c+1)*rk*(1-rk)/(self.a*(self.c-self.a))))
        UB = (rk*self.c-self.a)/(rk*(1-rk))*(1+rk/self.a)

        def f(kappa):
            return ((self.a/self.c)*(np.exp(self.kummer_log(k=kappa,a=self.a+1,c=self.c+1)-self.kummer_log(k=kappa,a=self.a,c=self.c)))-rk)**2
        options={}
        options['xatol'] = tol
        if rk>self.a/self.c:
            kappa = minimize_scalar(f, bounds=[LB, B],options=options,method='bounded')['x']
            if kappa<LB or kappa>B:
                print('Probably a convergence problem for kappa')
                return
        elif rk<self.a/self.c:
            kappa = minimize_scalar(f, bounds=[B, UB],options=options,method='bounded')['x']
            if kappa<B or kappa>UB:
                print('Probably a convergence problem for kappa')
                return
        elif rk==self.a/self.c:
            kappa = 0
        else:
            raise ValueError("kappa could not be optimized")
        return mu,kappa
    
    def M_step(self,X):
        # n,p = X.shape
        if self.K!=1:
            beta = np.exp(self.log_density-self.logsum_density)
            self.update_pi(beta)
        else:
            beta = np.ones((1,X.shape[0]))

        for k in range(self.K):
            self.mu[k],self.kappa[k] = self.M_step_single_component(X=X,beta=beta[k],mu=self.mu[k],kappa=self.kappa[k])

class ACG(PCMMnumpyBaseModel):
    def __init__(self, p:int, rank:int=None, K:int=1, complex:bool=False, params:dict=None):
        super().__init__()

        self.K = K
        self.p = p
        if rank is None or rank==0: #rank zero means the fullrank version
            self.r = p
            self.distribution = 'ACG_fullrank'
        else: #the lowrank version can also be full rank
            self.r = rank
            self.distribution = 'ACG_lowrank'

        if complex:
            self.a = 1
            self.c = self.p
            self.distribution = 'Complex_'+self.distribution
        else:
            self.a = 0.5
            self.c = self.p/2
        
        # precompute log-surface area of the unit hypersphere
        self.logSA_sphere = loggamma(self.c) - np.log(2) -self.c* np.log(np.pi)

        self.flag_normalized_input_data = False #flag to check if the input data is normalized

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)
    
    def log_pdf_lowrank(self, X):
        _,S1,V1h = np.linalg.svd(self.M,full_matrices=False)
        log_det_D = np.sum(np.log(1/S1**2+1),axis=-1)+2*np.sum(np.log(S1),axis=-1)
        D_inv = (np.swapaxes(V1h.conj(),-2,-1)*(1/(1+S1**2))[:,None])@V1h

        XM = X[None,:,:].conj()@self.M #check the conj here and below
        v = 1-np.sum(XM@D_inv*XM.conj(),axis=-1) 

        log_pdf = self.logSA_sphere - self.a * np.atleast_1d(log_det_D)[:,None] - self.c * np.log(np.real(v))
        return log_pdf
    
    def log_pdf_fullrank(self, X):
        XtLX = np.real(np.sum(X.conj()@np.linalg.inv(self.Psi)*X,axis=-1)) #should be real
        log_det_L = np.atleast_1d(np.real(self.logdet(self.Psi)))[:,None] #should be real
        log_pdf = self.logSA_sphere - self.a * log_det_L -self.c*np.log(XtLX)
        return log_pdf

    def log_pdf(self, X):
        if not self.flag_normalized_input_data:
            if not np.allclose(np.linalg.norm(X,axis=1),1):
                raise ValueError("For the ACG distribution, the input data vectors should be normalized to unit length.")
            else:
                self.flag_normalized_input_data = True
        if 'lowrank' in self.distribution:
            return self.log_pdf_lowrank(X)
        elif 'fullrank' in self.distribution:
            return self.log_pdf_fullrank(X)

    def M_step_single_component(self,X,beta,M=None,Lambda=None,max_iter=int(1e5),tol=1e-10):
        n,p = X.shape
        if n<p*(p-1):
            Warning("Too high dimensionality compared to number of observations. Lambda cannot be calculated")
        if self.distribution in ['ACG_lowrank','Complex_ACG_lowrank']:
            Q = (beta[:,None]*X).T

            loss = []
            o = np.linalg.norm(M,'fro')**2
            gamma = 1/(1+o/self.p)
            # M_tilde = M*np.sqrt(gamma)
            M = np.sqrt(gamma)*M #we control the scale of M to avoid numerical issues
            M_tilde = M
            S1 = np.linalg.svd(M_tilde,full_matrices=False,compute_uv=False)
            
            trZt_oldZ_old = np.sum(S1**4)+2*gamma*np.sum(S1**2)+gamma**2*self.p
            gamma_old = gamma
            S1_old = S1
            # M_old = M.copy()
            M_tilde_old = M_tilde.copy()

            for j in range(max_iter):

                # First we update M
                D_inv = np.linalg.inv(M.conj().T@M+np.eye(self.r))
                # D_inv = V1h.conj().T@np.diag(1/(1+S1**2))@V1h
                XM = X.conj()@M 
                XMD_inv = XM@D_inv
                v = 1-np.sum(XMD_inv*XM.conj(),axis=1) #denominator
                M = self.p/np.sum(beta)*Q@(XMD_inv/v[:,None])

                # Then we trace-normalize M
                o = np.linalg.norm(M,'fro')**2
                gamma = 1/(1+o/self.p)

                # To measure convergence, we compute norm(Z-Z_old)**2
                # M_tilde = M*np.sqrt(gamma)
                M = np.sqrt(gamma)*M
                M_tilde = M
                S1 = np.linalg.svd(M_tilde,full_matrices=False,compute_uv=False)
                trZtZ = np.sum(S1**4)+2*gamma*np.sum(S1**2)+gamma**2*self.p
                trZtZt_old = np.linalg.norm(M_tilde.conj().T@M_tilde_old)**2 + gamma_old*gamma*self.p + gamma_old*np.sum(S1**2) + gamma*np.sum(S1_old**2)
                loss.append(trZtZ+trZt_oldZ_old-2*trZtZt_old)
                
                if j>0:
                    if loss[-1]<tol:
                        if np.any(np.array(loss)<-tol):
                            raise Warning("Loss is negative. Check M_step_single_component")
                        break
                
                # # # To measure convergence
                trZt_oldZ_old = trZtZ
                gamma_old = gamma
                # M_old = M
                S1_old = S1
                M_tilde_old = M_tilde

            if j==max_iter-1:
                raise Warning("M-step did not converge")

            # output the unnormalized M such that the log_pdf is computed without having to include the normalization factor (pdf is scale invariant)
            M = M/np.sqrt(gamma)
            # print(j,np.linalg.norm(M),gamma,np.sum(beta))
            return M
        elif self.distribution in ['ACG_fullrank','Complex_ACG_fullrank']:
            Lambda_old = Lambda.copy()
            Q = (beta[:,None]*X).T
            for j in range(max_iter):
                XtLX = np.sum(X.conj()@np.linalg.inv(Lambda)*X,axis=1)
                Lambda = self.p/np.sum(beta/XtLX)*(Q/XtLX)@X.conj()
                if j>0:
                    if np.linalg.norm(Lambda_old-Lambda)<tol:
                        break
                Lambda_old = Lambda
            return Lambda

    def M_step(self,X):
        if self.K!=1:
            beta = np.exp(self.log_density-self.logsum_density)
            self.update_pi(beta)
        else:
            beta = np.ones((1,X.shape[0]))
        for k in range(self.K):
            if self.distribution in ['ACG_lowrank','Complex_ACG_lowrank']:
                self.M[k] = self.M_step_single_component(X,beta[k],self.M[k],None)
            elif self.distribution in ['ACG_fullrank','Complex_ACG_fullrank']:
                self.Psi[k] = self.M_step_single_component(X,beta[k],None,self.Psi[k])

class MACG(PCMMnumpyBaseModel):
    def __init__(self, p:int,q:int, K:int=1, rank=None, params:dict=None):
        super().__init__()

        self.K = K
        self.p = p
        self.q = q
        if rank is None or rank==0: #rank zero means the fullrank version
            self.r = p
            self.distribution = 'MACG_fullrank'
        else: #the lowrank version can also be full rank
            self.r = rank
            self.distribution = 'MACG_lowrank'

        self.flag_normalized_input_data = False #flag to check if the input data is normalized

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)
    
    def log_pdf_lowrank(self, X):

        _,S1,V1t = np.linalg.svd(self.M,full_matrices=False)
        log_det_D = np.sum(np.log(1/S1**2+1),axis=-1)+2*np.sum(np.log(S1),axis=-1)

        D_sqrtinv = (np.swapaxes(V1t,-2,-1)*np.sqrt(1/(1+S1**2))[:,None])@V1t
        XtM = np.swapaxes(X,-2,-1)[None,:,:,:]@self.M[:,None,:,:]
        S2 = np.linalg.svd(XtM@(D_sqrtinv[:,None]),compute_uv=False)
        v = np.sum(np.log(1/(S2**2)-1),axis=-1)+2*np.sum(np.log(S2),axis=-1)
        
        log_pdf = - (self.q/2)*np.atleast_1d(log_det_D)[:,None] - self.p/2*v
        return log_pdf

    def log_pdf_fullrank(self, X):
        logdetXtLX = self.logdet(np.swapaxes(X,-2,-1)[None,:,:,:]@np.linalg.inv(self.Psi)[:,None,:,:]@X)
        log_pdf = - (self.q/2)*np.atleast_1d(self.logdet(self.Psi))[:,None] - self.p/2*logdetXtLX
        return log_pdf

    def log_pdf(self, X):
        if not self.flag_normalized_input_data:
            if np.allclose(np.linalg.norm(X[:,:,0],axis=1),1)!=1:
                raise ValueError("For the MACG distribution, the input data vectors should be normalized to unit length (and orthonormal, but this is not checked).")
            else:
                self.flag_normalized_input_data = True
        if self.distribution == 'MACG_lowrank':
            return self.log_pdf_lowrank(X)
        elif self.distribution == 'MACG_fullrank':
            return self.log_pdf_fullrank(X)

    def M_step_single_component(self,X,beta,M=None,Lambda=None,max_iter=int(1e5),tol=1e-10):
            
        if self.distribution == 'MACG_lowrank':
            Q = beta[:,None,None]*X

            loss = []
            o = np.linalg.norm(M,'fro')**2
            gamma = 1/(1+o/self.p)
            M = np.sqrt(gamma)*M
            # M_tilde = M*np.sqrt(gamma)
            M_tilde = M
            S1 = np.linalg.svd(M_tilde,full_matrices=False,compute_uv=False)
            
            trZt_oldZ_old = np.sum(S1**4)+2*gamma*np.sum(S1**2)+gamma**2*self.p
            gamma_old = gamma
            S1_old = S1
            # M_old = M.copy()
            M_tilde_old = M_tilde.copy()

            for j in range(max_iter):
                D_sqrtinv = sqrtm(np.linalg.inv(M.conj().T@M+np.eye(self.r)))
                # D_sqrtinv = V1t.T@np.diag(np.sqrt(1/(1+S1**2)))@V1t
                U2,S2,V2t = np.linalg.svd(np.swapaxes(X,-2,-1)@(M@D_sqrtinv),full_matrices=False)
                M = self.p/(self.q*np.sum(beta))*np.sum(Q@(U2*(S2/(1-S2**2))[:,None,:])@V2t,axis=0)@D_sqrtinv

                # Then we trace-normalize M
                o = np.linalg.norm(M,'fro')**2
                gamma = 1/(1+o/self.p)
                M = np.sqrt(gamma)*M

                # To measure convergence, we compute norm(Z-Z_old)**2
                # M_tilde = M*np.sqrt(gamma)
                M_tilde = M
                S1 = np.linalg.svd(M_tilde,full_matrices=False,compute_uv=False)
                trZtZ = np.sum(S1**4)+2*gamma*np.sum(S1**2)+gamma**2*self.p
                trZtZt_old = np.linalg.norm(M_tilde.T@M_tilde_old)**2 + gamma_old*gamma*self.p + gamma_old*np.sum(S1**2) + gamma*np.sum(S1_old**2)
                loss.append(trZtZ+trZt_oldZ_old-2*trZtZt_old)
                # t4 = time()
                # print(j,t1-t0,t2-t1,t3-t2,t4-t3)
                if j>0:
                    if loss[-1]<tol:
                        if np.any(np.array(loss)<-tol):
                            raise Warning("Loss is negative. Check M_step_single_component")
                        break
                
                # To measure convergence
                trZt_oldZ_old = trZtZ
                gamma_old = gamma
                # M_old = M
                M_tilde_old = M_tilde
                S1_old = S1

            if j==max_iter-1:
                raise Warning("M-step did not converge")
            
            # output the unnormalized M such that the log_pdf is computed without having to include the normalization factor (pdf is scale invariant)
            M = M/np.sqrt(gamma)
            # print(j,np.linalg.norm(M),gamma,np.sum(beta))
            return M
        elif self.distribution == 'MACG_fullrank':
            Q = beta[:,None,None]*X
            Lambda_old = Lambda.copy()

            for j in range(max_iter):

                # this has been tested in the "Naive" version below
                XtLX = np.swapaxes(X,-2,-1)@np.linalg.inv(Lambda)@X
                L = np.linalg.eigvalsh(XtLX)
                XtLX_trace = np.sum(1/L,axis=1) #trace of inverse is sum of inverse eigenvalues
                
                Lambda = self.p*np.sum(Q@np.linalg.inv(XtLX)@np.swapaxes(X,-2,-1),axis=0) \
                    /(np.sum(beta*XtLX_trace))
                
                if j>0:
                    if np.linalg.norm(Lambda_old-Lambda)<tol:
                        break
                Lambda_old = Lambda
            return Lambda

    def M_step(self,X):
        if self.K!=1:
            beta = np.exp(self.log_density-self.logsum_density)
            self.update_pi(beta)
        else:
            beta = np.ones((1,X.shape[0]))
        for k in range(self.K):
            if self.distribution == 'MACG_lowrank':
                self.M[k] = self.M_step_single_component(X,beta[k],self.M[k],None)
            elif self.distribution == 'MACG_fullrank':
                self.Psi[k] = self.M_step_single_component(X,beta[k],None,self.Psi[k])

class SingularWishart(PCMMnumpyBaseModel):
    def __init__(self, p:int,q:int, K:int=1, rank=None, params:dict=None):
        super().__init__()

        self.K = K
        self.p = p
        self.q = q
        self.half_p = self.p/2
        if rank is None or rank==0: #rank zero means the fullrank version
            self.r = p
            self.distribution = 'SingularWishart_fullrank'
        else: #the lowrank version can also be full rank
            self.r = rank
            self.distribution = 'SingularWishart_lowrank'

        loggamma_k = (self.q*(self.q-1)/4)*np.log(np.pi)+np.sum(loggamma(self.q-np.arange(self.q)/2))
        self.log_norm_constant = self.q*(self.q-self.p)/2*np.log(np.pi)-self.p*self.q/2*np.log(2)-loggamma_k

        self.log_det_S11 = None

        self.flag_normalized_input_data = False #flag to check if the input data is normalized

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)
    
    def log_pdf_lowrank(self, X):

        M_tilde = self.M*np.sqrt(1/np.atleast_1d(self.gamma)[:,None,None])
        
        _,S1,V1t = np.linalg.svd(M_tilde)
        log_det_D = self.p*np.log(self.gamma)+np.sum(np.log(1/S1**2+1),axis=-1)+2*np.sum(np.log(S1),axis=-1)
        D_sqrtinv = (np.swapaxes(V1t,-2,-1)*np.sqrt(1/(1+S1**2))[:,None])@V1t

        # while Q_q^T Q_q != U_q^T L U_q, their determinants are the same
        if self.log_det_S11 is None:
            self.log_det_S11 = self.logdet(np.swapaxes(X[:,:self.q,:],-2,-1)@X[:,:self.q,:])
        
        v = 1/np.atleast_1d(self.gamma)[:,None]*(self.p - np.linalg.norm(np.swapaxes(X,-2,-1)[None,:,:,:]@M_tilde[:,None,:,:]@D_sqrtinv[:,None],axis=(-2,-1))**2)
        log_pdf = self.log_norm_constant - (self.q/2)*np.atleast_1d(log_det_D)[:,None] + (self.q-self.p-1)/2*self.log_det_S11[None] - 1/2*v

        return log_pdf

    def log_pdf_fullrank(self, X):
        log_pdf = self.log_norm_constant - (self.q/2)*np.atleast_1d(self.logdet(self.Psi))[:,None] - 1/2*np.trace(np.swapaxes(X,-2,-1)[None,:]@np.linalg.inv(self.Psi)[:,None]@X[None,:],axis1=-2,axis2=-1)
        return log_pdf

    def log_pdf(self, X):
        if not self.flag_normalized_input_data:
            X_weights = np.linalg.norm(X,axis=1)**2
            if not np.allclose(np.sum(X_weights,axis=1),self.p):
                raise ValueError("In weighted grassmann clustering, the scale of the input data vectors should be equal to the square root of the eigenvalues. If the scale does not sum to the dimensionality, this error is thrown")
            else:
                self.flag_normalized_input_data = True
        if self.distribution == 'SingularWishart_lowrank':
            return self.log_pdf_lowrank(X)
        elif self.distribution == 'SingularWishart_fullrank':
            return self.log_pdf_fullrank(X)
        
    def M_step_single_component(self,X,beta,M=None,gamma=None,max_iter=int(1e8),tol=1e-10):
            
        if self.distribution == 'SingularWishart_lowrank':

            loss = []
            
            if gamma is None:
                raise ValueError("gamma is not provided")
            
            M_tilde = M * np.sqrt(1/gamma)
            _,S1,V1t = np.linalg.svd(M_tilde)
            D_inv = V1t.T@np.diag(1/(1+S1**2))@V1t

            trZt_oldZ_old = gamma**2*(np.sum(S1**4)+2*np.sum(S1**2)+self.p)
            gamma_old = gamma
            S1_old = S1
            M_tilde_old = M_tilde

            #precompute
            beta_sum = np.sum(beta)
            beta_S = np.sum(beta[:,None,None]*X@np.swapaxes(X,-2,-1),axis=0)

            for j in range(max_iter):

                # M_tilde update
                # M_tilde = 1/(gamma*self.q*beta_sum)*np.sum(beta_Q*QtM_tilde[:,None,:,:],axis=(0,-2))@D_inv
                M_tilde = 1/(gamma*self.q*beta_sum)*beta_S@M_tilde@D_inv
                # _,S1,V1t = np.linalg.svd(M_tilde)
                # D_inv = V1t.T@np.diag(1/(1+S1**2))@V1t
                S1 = np.linalg.svd(M_tilde,full_matrices=False,compute_uv=False)
                D_inv = np.linalg.inv(M_tilde.conj().T@M_tilde+np.eye(self.r))

                #gamma update
                # QtM_tilde = np.swapaxes(Q,-2,-1)@M_tilde
                # gamma = 1/(self.q*self.p*beta_sum)*np.sum(beta*(self.p-np.linalg.norm(QtM_tilde@V1t.T@np.diag(np.sqrt(1/(1+S1**2)))@V1t,axis=(-2,-1))**2))
                gamma = 1/(self.q*self.p*beta_sum)*(np.sum(beta*self.p)-np.trace(M_tilde@D_inv@M_tilde.T@beta_S))

                # convergence criterion
                trZtZ = gamma**2*(np.sum(S1**4)+2*np.sum(S1**2)+self.p)
                trZtZt_old = gamma*gamma_old*(np.linalg.norm(M_tilde.T@M_tilde_old)**2 + np.sum(S1**2) + np.sum(S1_old**2)+self.p)
                loss.append(trZtZ+trZt_oldZ_old-2*trZtZt_old)
                
                if j>0:
                    if loss[-1]<tol:
                        if np.any(np.array(loss)<-tol):
                            raise Warning("Loss is negative. Check M_step_single_component")
                        break
                
                # To measure convergence
                trZt_oldZ_old = trZtZ
                gamma_old = gamma
                M_tilde_old = M_tilde
                S1_old = S1

            if j==max_iter-1:
                raise Warning("M-step did not converge")

            # output M, not M_tilde
            M = M_tilde*np.sqrt(gamma)
            # print(j,np.linalg.norm(M),gamma,np.sum(beta))
            return M, gamma
        elif self.distribution == 'SingularWishart_fullrank':
            Psi = 1/(self.q*np.sum(beta))*np.sum((beta[:,None,None]*X)@np.swapaxes(X,-2,-1),axis=0)
            return Psi

    def M_step(self,X):
        if self.K!=1:
            beta = np.exp(self.log_density-self.logsum_density)
            self.update_pi(beta)
        else:
            beta = np.ones((1,X.shape[0]))
        for k in range(self.K):
            if self.distribution == 'SingularWishart_lowrank':
                self.M[k],self.gamma[k] = self.M_step_single_component(X,beta[k],self.M[k],gamma=self.gamma[k])
            elif self.distribution == 'SingularWishart_fullrank':
                self.Psi[k] = self.M_step_single_component(X,beta[k])

class Normal(PCMMnumpyBaseModel):
    def __init__(self, p:int, rank:int=None, K:int=1, complex:bool=False, params:dict=None):
        super().__init__()

        self.K = K
        self.p = p
        if rank is None or rank==0: #rank zero means the fullrank version
            self.r = p
            self.distribution = 'Normal_fullrank'
        else: #the lowrank version can also be full rank
            self.r = rank
            self.distribution = 'Normal_lowrank'

        if complex:
            self.a = 1
            self.c = self.p
            self.distribution = 'Complex_'+self.distribution
        else:
            self.a = 0.5
            self.c = self.p/2
        
        self.log_norm_constant = -self.c*np.log(1/self.a*np.pi)
        self.norm_x = None

        # initialize parameters
        if params is not None:            
            self.unpack_params(params)
    
    def log_pdf_lowrank(self, X):

        M_tilde = self.M*np.sqrt(1/np.atleast_1d(self.gamma)[:,None,None])
        
        _,S1,V1t = np.linalg.svd(M_tilde)
        log_det_D = self.p*np.log(self.gamma)+np.sum(np.log(1/S1**2+1),axis=-1)+2*np.sum(np.log(S1),axis=-1)
        D_sqrtinv = (np.swapaxes(V1t.conj(),-2,-1)*np.sqrt(1/(1+S1**2))[:,None])@V1t

        if self.norm_x is None:
            self.norm_x = np.linalg.norm(X,axis=1)**2
        
        v = 1/np.atleast_1d(self.gamma)[:,None]*(self.norm_x[None,:] - np.linalg.norm(X.conj()[None,:,None,:]@M_tilde[:,None,:,:]@D_sqrtinv[:,None,:,:],axis=(-2,-1))**2)
        log_pdf = self.log_norm_constant - self.a*np.atleast_1d(log_det_D)[:,None] - self.a*np.real(v)

        return log_pdf

    def log_pdf_fullrank(self, X):
        XtLX = np.real(np.sum(X.conj()@np.linalg.inv(self.Psi)*X,axis=-1)) #should be real
        log_det_L = np.atleast_1d(np.real(self.logdet(self.Psi)))[:,None] #should be real
        log_pdf = self.log_norm_constant - self.a * log_det_L -self.a*XtLX
        return log_pdf

    def log_pdf(self, X):
        if 'lowrank' in self.distribution:
            return self.log_pdf_lowrank(X)
        elif 'fullrank' in self.distribution:
            return self.log_pdf_fullrank(X)
        
    def M_step_single_component(self,X,beta,M=None,gamma=None,max_iter=int(1e8),tol=1e-10):
            
        if 'lowrank' in self.distribution:

            loss = []
            
            if gamma is None:
                raise ValueError("gamma is not provided")

            if self.norm_x is None:
                norm_x = np.linalg.norm(X,axis=1)**2
            else:
                norm_x = self.norm_x
            
            M_tilde = M * np.sqrt(1/gamma)
            # _,S1,V1t = np.linalg.svd(M_tilde)
            S1 = np.linalg.svd(M_tilde,full_matrices=False,compute_uv=False)
            # D_inv = V1t.T.conj()@np.diag(1/(1+S1**2))@V1t
            D_inv = np.linalg.inv(M_tilde.conj().T@M_tilde+np.eye(self.r))

            trZt_oldZ_old = gamma**2*(np.sum(S1**4)+2*np.sum(S1**2)+self.p)
            gamma_old = gamma
            S1_old = S1
            M_tilde_old = M_tilde

            #precompute
            beta_sum = np.sum(beta)
            beta_S = (beta[:,None]*X).T@X.conj()
            beta_x_norm_sum = np.sum(beta*norm_x)

            for j in range(max_iter):

                # M_tilde update
                M_tilde = 1/(gamma*beta_sum)*beta_S@M_tilde@D_inv
                # _,S1,V1t = np.linalg.svd(M_tilde)
                S1 = np.linalg.svd(M_tilde,full_matrices=False,compute_uv=False)
                # D_inv = V1t.T.conj()@np.diag(1/(1+S1**2))@V1t
                D_inv = np.linalg.inv(M_tilde.conj().T@M_tilde+np.eye(self.r))
                #gamma update
                gamma = 1/(self.p*beta_sum)*(beta_x_norm_sum-np.trace(M_tilde@D_inv@M_tilde.T.conj()@beta_S).real)

                # convergence criterion
                trZtZ = gamma**2*(np.sum(S1**4)+2*np.sum(S1**2)+self.p)
                trZtZt_old = gamma*gamma_old*(np.linalg.norm(M_tilde.T.conj()@M_tilde_old)**2 + np.sum(S1**2) + np.sum(S1_old**2)+self.p)
                loss.append(trZtZ+trZt_oldZ_old-2*trZtZt_old)
                
                if j>0:
                    if loss[-1]<tol:
                        # if np.any(np.array(loss)<tol):
                            # if loss[-1]<-1e-5:
                            #     raise Warning("Loss is negative. Check M_step_single_component")
                            # raise Warning("Loss is negative. Check M_step_single_component")
                        break
                
                # To measure convergence
                trZt_oldZ_old = trZtZ
                gamma_old = gamma
                M_tilde_old = M_tilde
                S1_old = S1

            if j==max_iter-1:
                raise Warning("M-step did not converge")

            # output M, not M_tilde
            M = M_tilde*np.sqrt(gamma)
            # print(j,np.linalg.norm(M),gamma,np.sum(beta))
            return M, gamma
        elif 'fullrank' in self.distribution:
            Psi = 1/(np.sum(beta))*(beta[:,None]*X).T@X
            return Psi

    def M_step(self,X):
        if self.K!=1:
            beta = np.exp(self.log_density-self.logsum_density)
            self.update_pi(beta)
        else:
            beta = np.ones((1,X.shape[0]))
        for k in range(self.K):
            if 'lowrank' in self.distribution:
                self.M[k],self.gamma[k] = self.M_step_single_component(X,beta[k],self.M[k],gamma=self.gamma[k])
            elif 'fullrank' in self.distribution:
                self.Psi[k] = self.M_step_single_component(X,beta[k])