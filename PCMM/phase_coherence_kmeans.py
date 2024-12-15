import numpy as np
import scipy
from scipy.cluster.vq import kmeans2

def plusplus_initialization(X,K,dist='diametrical'):
    assert dist in ['diametrical','grassmann','weighted_grassmann']
    n = X.shape[0]
    if X.ndim == 3:
        q = X.shape[2]

    # choose first centroid at random from X
    idx = np.random.choice(n,p=None)

    if dist == 'diametrical':
        C = X[idx][:,np.newaxis]
    else:
        C = np.zeros((K,X.shape[1],X.shape[2]))
        C[0] = X[idx]
        if dist == 'weighted_grassmann':
            X_weights = np.linalg.norm(X,axis=1)**2
            # X = X/np.linalg.norm(X,axis=1)[:,None,:]
            C_weights = np.zeros((K,X.shape[2]))
            C_weights[0] = X_weights[idx]
            X_weights = np.delete(X_weights,idx,axis=0)
    
    X = np.delete(X,idx,axis=0)

    # for all other centroids, compute the distance from all X to the current set of centroids. 
    # Construct a weighted probability distribution and sample using this. 

    for k in range(K):
        if dist == 'diametrical':
            dis = 1-np.abs(X@C.conj())**2  #abs for complex support
            dis = np.clip(dis,0,None)
        elif dist == 'grassmann':
            dis = 1/np.sqrt(2)*(2*q-2*np.linalg.norm(np.swapaxes(X[:,None],-2,-1)@C[:k+1][None],axis=(-2,-1)))
            dis = np.clip(dis,0,None)
        elif dist == 'weighted_grassmann':
            B = np.swapaxes(X,-2,-1)[:,None]@(C[:k+1][None])
            dis = 1/np.sqrt(2)*(np.sum(X_weights**2,axis=1)[:,None]+np.sum(C_weights[:k+1]**2,axis=1)[None]-2*np.linalg.norm(B,axis=(-2,-1))**2)#
        
        mindis = np.min(dis,axis=1) #choose the distance to the closest centroid for each point

        if k==K-1:
            X_part = np.argmin(dis,axis=1)
            obj = np.mean(np.max(-dis,axis=1))
            break

        prob_dist = mindis/np.sum(mindis) # construct the prob. distribution
        idx = np.random.choice(n-k-1,p=prob_dist)
        if dist == 'diametrical':
            C = np.hstack((C,X[idx][:,np.newaxis]))
        else:
            C[k+1] = X[idx]
        X = np.delete(X,idx,axis=0)
        if dist == 'weighted_grassmann':
            C_weights[k+1] = X_weights[idx]
            X_weights = np.delete(X_weights,idx,axis=0)
    if dist == 'weighted_grassmann':
        return C/np.linalg.norm(C,axis=1)[:,None,:],C_weights,X_part,obj
    else:
        return C,X_part,obj
    
def diametrical_clustering(X,K,max_iter=10000,num_repl=1,init=None,tol=1e-10):

    if not np.allclose(np.linalg.norm(X,axis=1),1):
        raise ValueError("In diametrical clustering, the input data vectors should be normalized to unit length.")

    n,p = X.shape

    obj_collector = []
    obj_final_collector = []
    part_collector = []
    C_collector = []

    for _ in range(num_repl):
        if init is None or init=='++' or init=='plusplus' or init == 'diametrical_clustering_plusplus':
            C,_,_ = plusplus_initialization(X,K,dist='diametrical')
        elif init=='uniform' or init=='unif':
            if X.dtype == 'complex':
                C = np.random.uniform(size=(p,K))+1j*np.random.uniform(size=(p,K))
            else:
                C = np.random.uniform(size=(p,K))
            C = C/np.linalg.norm(C,axis=0)
        
        iter = 0
        obj = []
        partsum = np.zeros((max_iter,K))
        while True:
            # E-step
            sim = np.abs(X@C.conj())**2 #abs for complex support
            maxsim = np.max(sim,axis=1)
            X_part = np.argmax(sim,axis=1)
            obj.append(np.mean(maxsim))

            for k in range(K):
                partsum[iter,k] = np.sum(X_part==k)
            
            if iter>0:
                if all((partsum[iter-1]-partsum[iter])==0) or iter==max_iter or obj[-1]-obj[-2]<tol:
                    C_collector.append(C)
                    obj_collector.append(obj)
                    obj_final_collector.append(obj[-1])
                    part_collector.append(X_part)             
                    break
            
            for k in range(K):
                idx_k = X_part==k
                # if sum(idx_k)==0:
                #     C[:,k] = np.random.uniform(size=p)
                #     continue
                
                # # Establish covariance matrix
                A = X[idx_k].T@X[idx_k].conj()
                C[:,k] = A@C[:,k]

            C = C/np.linalg.norm(C,axis=0)
            iter += 1
    best = np.nanargmax(np.array(obj_final_collector))
    
    return C_collector[best],part_collector[best],obj_collector[best]

def grassmann_clustering(X,K,max_iter=10000,num_repl=1,init=None,tol=1e-10):
    
    if np.allclose(np.linalg.norm(X[:,:,0],axis=1),1)!=1:
        raise ValueError("In grassmann clustering, the input data vectors should be normalized to unit length.")

    n,p,q = X.shape

    obj_collector = [] # objective function collector
    obj_final_collector = [] # final objective function collector
    part_collector = [] # partition collector
    C_collector = [] # cluster center collector

    # loop over the number of repetitions
    for _ in range(num_repl):
        # initialize cluster centers
        if init is None or init in ['++','plusplus','grassmann_clustering_plusplus']:
            C,_,_ = plusplus_initialization(X,K,dist='grassmann')
        elif init in ['unif','uniform']:
            C = np.random.uniform(size=(K,p,q))
            for k in range(K):
                C[k] = C[k]@scipy.linalg.sqrtm(np.linalg.inv(C[k].T@C[k])) # project onto the Grassmannian

        iter = 0
        obj = [] # objective function
        partsum = np.zeros((max_iter,K))
        while True:

            dis = 1/np.sqrt(2)*(2*q-2*np.linalg.norm(np.swapaxes(X[:,None],-2,-1)@C[None],axis=(-2,-1))**2)
            sim = -dis

            maxsim = np.max(sim,axis=1) # find the maximum similarity
            X_part = np.argmax(sim,axis=1) # assign each point to the cluster with the highest similarity
            obj.append(np.mean(maxsim))

            # check for convergence
            for k in range(K):
                partsum[iter,k] = np.sum(X_part==k)
            
            if iter>0:
                if all((partsum[iter-1]-partsum[iter])==0) or iter==max_iter or obj[-1]-obj[-2]<tol:
                    C_collector.append(C)
                    obj_collector.append(obj)
                    obj_final_collector.append(obj[-1])
                    part_collector.append(X_part)             
                    break
            
            # "M-step" - update the cluster centers
            for k in range(K):
                idx_k = X_part==k
                V = np.reshape(np.swapaxes(X[idx_k],0,1),(p,np.sum(idx_k)*q))
                U,S,_ = scipy.sparse.linalg.svds(V,q,return_singular_vectors="u")
                order = np.argsort(S)[::-1]
                C[k] = U[:,order]#[:,::-1]
                # C[k] = U[:,:q]
            iter += 1
    best = np.nanargmax(np.array(obj_final_collector))
    return C_collector[best],part_collector[best],obj_collector[best]

def weighted_grassmann_clustering(X,K,max_iter=10000,num_repl=1,tol=1e-10,init=None):
    """"
    Weighted grassmannian clustering using the chordal distance function and a SVD-based update rule
    
    X: size (nxpxq), where n is the number of observations, p is the number of features and q is the subspace dimensionality
    X_weights: size (n,q), where n is the number of observations and q is the subspace dimensionality (corresponds to eigenvalues)
    K: number of clusters
    max_iter: maximum number of iterations
    tol: tolerance for convergence

    """
    
    n,p,q = X.shape
    # Q = X*np.sqrt(X_weights[:,None,:])
    X_weights = np.linalg.norm(X,axis=1)**2
    if not np.allclose(np.sum(X_weights,axis=1),p):
        raise ValueError("In weighted grassmann clustering, the scale of the input data vectors should be equal to the square root of the eigenvalues. If the scale does not sum to the dimensionality, this error is thrown")

    obj_collector = [] # objective function collector
    obj_final_collector = [] # final objective function collector
    part_collector = [] # partition collector
    C_collector = [] # cluster center collector
    C_weights_collector = [] # cluster weights collector

    # loop over the number of repetitions
    for _ in range(num_repl):
        # initialize cluster centers
        if init is None or init=='++' or init=='plusplus' or init == 'weighted_grassmann_clustering_plusplus':
            C,C_weights,_,_ = plusplus_initialization(X,K,dist='weighted_grassmann')
        elif init=='uniform' or init=='unif':
            C = np.random.uniform(size=(K,p,q))
            C_weights = np.ones((K,q))*p/q
            for k in range(K):
                C[k] = C[k]@scipy.linalg.sqrtm(np.linalg.inv(C[k].T@C[k])) # project onto the Grassmannian

        # initialize counters
        iter = 0
        obj = []
        partsum = np.zeros((max_iter,K))
        while True:
            # "E-step" - compute the similarity between each matrix and each cluster center
            B = np.swapaxes(X,-2,-1)[:,None]@((C*np.sqrt(C_weights[:,None,:]))[None])
            dis = 1/np.sqrt(2)*(np.sum(X_weights**2,axis=-1)[:,None]+np.sum(C_weights**2,axis=-1)[None]-2*np.linalg.norm(B,axis=(-2,-1))**2)#
            sim = -dis
            maxsim = np.max(sim,axis=1) # find the maximum similarity - the sum of this value is the objective function
            X_part = np.argmax(sim,axis=1) # assign each point to the cluster with the highest similarity
            obj.append(np.mean(maxsim))

            # check for convergence   
            for k in range(K):
                partsum[iter,k] = np.sum(X_part==k)
            if iter>0:
                if all((partsum[iter-1]-partsum[iter])==0) or iter==max_iter or obj[-1]-obj[-2]<tol:
                    C_collector.append(C)
                    C_weights_collector.append(C_weights)
                    obj_collector.append(obj)
                    obj_final_collector.append(obj[-1])
                    part_collector.append(X_part)             
                    break
            
            # "M-step" - update the cluster centers
            for k in range(K):
                idx_k = X_part==k
                V = np.reshape(np.swapaxes(X[idx_k],0,1),(p,np.sum(idx_k)*q))
                U,S,_ = scipy.sparse.linalg.svds(V,q,return_singular_vectors="u")
                order = np.argsort(S)[::-1]
                C[k] = U[:,order]
                C_weights[k] = S[order]
                C_weights[k] = C_weights[k]/np.sum(C_weights[k])*p

            iter += 1
    best = np.nanargmax(np.array(obj_final_collector))
    return C_collector[best],C_weights_collector[best],part_collector[best],obj_collector[best]

def least_squares_sign_flip(X,K,max_iter=10000,num_repl=1,tol=1e-10,init=None):
    if init=='uniform':
        init = 'random'
    n,p = X.shape

    # perform the sign flip
    X[(X>0).sum(axis=1)>p/2] = -X[(X>0).sum(axis=1)>p/2]

    obj_collector = [] # objective function collector
    part_collector = [] # partition collector
    C_collector = [] # cluster center collector

    # loop over the number of repetitions
    for _ in range(num_repl):
        C,labels = kmeans2(X,k=K,minit=init,iter=max_iter)
        sim = -np.sum((X[:,None]-C[None])**2,axis=-1)
        obj = np.mean(np.max(sim,axis=1))

        C_collector.append(C.T)
        obj_collector.append(obj)
        part_collector.append(labels)   

    best = np.nanargmax(np.array(obj_collector))
    return C_collector[best],part_collector[best],obj_collector[best]        