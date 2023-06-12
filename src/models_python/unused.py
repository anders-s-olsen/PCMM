import numpy as np
from scipy.special import loggamma,hyp1f1
import time
import torch

class Watson_test():
    """
    Mixture model class
    """
    def __init__(self, K: int, p: int,init=None,num_kummer_eval=10000):
        super().__init__()
        self.K = K
        self.p = p
        self.n = np.arange(num_kummer_eval)
        self.log_kummer_tmp = loggamma(0.5 + self.n) + loggamma(self.p/2) - loggamma(0.5) - loggamma(self.p/2 + self.n) - loggamma(self.n + 1)

        
    def log_kummer_np(self, a, b, kappa,num_eval=10000):

        n = np.arange(num_eval)
    
        inner = loggamma(a + n) + loggamma(b) - loggamma(a) - loggamma(b + n) \
                + n * np.log(kappa) - loggamma(n + 1)
    
        logkum = np.logaddexp.reduce(inner, axis=0)
    
        return logkum
    def log_kummer_np2(self,kappa):
    
        inner = self.log_kummer_tmp + self.n * np.log(kappa)
    
        logkum = np.logaddexp.reduce(inner, axis=0)
    
        return logkum

    def log_kummer_torch(self, a, b, kappa,num_eval=10000):

        n = torch.arange(num_eval)
    
        inner = torch.lgamma(a + n) + torch.lgamma(b) - torch.lgamma(a) - torch.lgamma(b + n) \
                + n * torch.log(kappa) - torch.lgamma(n + torch.tensor(1.))
    
        logkum = torch.logsumexp(inner, dim=0)
    
        return logkum
    
    def log_sum(self,x, y, sign):
        s = x + np.log(1 + sign * np.exp(y - x))
        return s

    def kummer_log(self,a, c, k, n=10000):
        N = np.size(k)
        logkum = np.zeros(N)
        logkum_old = np.ones(N)
        tol = 1e-10
        foo = np.zeros(N)
        j = 1
        while np.any(np.abs(logkum - logkum_old) > tol) and (j < n):
            logkum_old = logkum
            foo += np.log((a + j - 1) / (j * (c + j - 1)) * k)
            logkum = self.log_sum(logkum, foo, 1)
            j += 1
        return logkum


if __name__=='__main__':
    K = np.array(2)
    p = np.array(92000)
    W = Watson_test(K=K,p=p)

    # Time function 1
    start_time = time.time()
    logkum1=W.log_kummer_np(np.array(0.5),p/2,np.array(1200))
    end_time = time.time()
    execution_time1 = end_time - start_time
    print("Execution time of function 1:", execution_time1)

    # Time function 2
    start_time = time.time()
    logkum2=W.log_kummer_torch(torch.tensor(0.5),torch.tensor(10/2),torch.tensor(1200))
    end_time = time.time()
    execution_time2 = end_time - start_time
    print("Execution time of function 2:", execution_time2)

    # Time function 3
    start_time = time.time()
    logkum3=W.kummer_log(np.array(0.5),p/2,np.array(1200))
    end_time = time.time()
    execution_time3 = end_time - start_time
    print("Execution time of function 3:", execution_time3)

    # Time function 4
    start_time = time.time()
    logkum4=W.log_kummer_np2(np.array(1200))
    end_time = time.time()
    execution_time4 = end_time - start_time
    print("Execution time of function 4:", execution_time4)

    # Time function 5
    start_time = time.time()
    logkum5=np.log(hyp1f1(0.5,p/2,1200))
    end_time = time.time()
    execution_time5 = end_time - start_time
    print("Execution time of function 5:", execution_time5)

    stop=7