import numpy as np
import scipy

def diametrical_clustering(X,K,max_iter=10000,num_repl=1,init=None,call=0,tol=1e-16):
    n,p = X.shape

    obj_final = []
    part_final = []
    C_final = []

    for _ in range(num_repl):
        if init is None or init=='++' or init=='plusplus' or init == 'diametrical_clustering_plusplus':
            C,_,_ = diametrical_clustering_plusplus(X,K)
        elif init=='uniform' or init=='unif':
            C = np.random.uniform(size=(p,K))
            C = C/np.linalg.norm(C,axis=0)
        
        iter = 0
        obj = []
        partsum = np.zeros((max_iter,K))
        while True:
            # E-step
            dis = (X@C)**2
            maxdis = np.max(dis,axis=1) # check that this works
            X_part = np.argmax(dis,axis=1)
            obj.append(np.mean(maxdis))

            for k in range(K):
                partsum[iter,k] = np.sum(X_part==k)
            
            if iter>0:
                if all((partsum[iter-1]-partsum[iter])==0) or iter==max_iter or obj[-2]-obj[-1]<tol:
                    C_final.append(C)
                    obj_final.append(obj[-1])
                    part_final.append(X_part)             
                    break
            
            for k in range(K):
                idx_k = X_part==k
                # Establish covariance matrix
                A = X[idx_k].T@X[idx_k]
                C[:,k] = A@C[:,k]
            C = C/np.linalg.norm(C,axis=0)
            iter += 1
    best = np.nanargmax(np.array(obj_final))
    
    return C_final[best],part_final[best],obj_final[best]

def diametrical_clustering_plusplus(X,K):
    n,_ = X.shape

    # choose first centroid at random from X
    idx = np.random.choice(n,p=None)
    C = X[idx][:,np.newaxis]
    X = np.delete(X,idx,axis=0)

    # for all other centroids, compute the distance from all X to the current set of centroids. 
    # Construct a weighted probability distribution and sample using this. 

    for k in range(K-1):
        dist = 1-(X@C)**2 #large means far away
        min_dist = np.min(dist,axis=1) #choose the distance to the closest centroid for each point
        prob_dist = min_dist/np.sum(min_dist) # construct the prob. distribution
        idx = np.random.choice(n-k-1,p=prob_dist)
        C = np.hstack((C,X[idx][:,np.newaxis]))
        X = np.delete(X,idx,axis=0)
    
    dis = (X@C)**2
    maxdis = np.max(dis,axis=1) # check that this works
    X_part = np.argmax(dis,axis=1)
    obj = np.mean(maxdis)
    return C,X_part,obj

def grassmannian_clustering(X,K,max_iter=10000,num_repl=1,init=None,call=0,tol=1e-16):
    
    n,p,q = X.shape

    obj_final = [] # objective function collector
    part_final = [] # partition collector
    C_final = [] # cluster center collector

    # loop over the number of repetitions
    for _ in range(num_repl):
        # initialize cluster centers
        C = np.random.uniform(size=(K,p,q))
        for k in range(K):
            C[k] = C[k]@scipy.linalg.sqrtm(np.linalg.inv(C[k].T@C[k])) # project onto the Grassmannian

        iter = 0
        obj = [] # objective function
        partsum = np.zeros((max_iter,K))
        while True:

            dis = 1/np.sqrt(2)(2*q-2*np.linalg.norm(np.swapaxes(X[:,None],-2,-1)@C[None],axis=(-2,-1)))
            sim = -dis

            maxsim = np.max(sim,axis=1) # find the maximum similarity
            X_part = np.argmax(sim,axis=1) # assign each point to the cluster with the highest similarity
            obj.append(np.mean(maxsim))

            # check for convergence
            for k in range(K):
                partsum[iter,k] = np.sum(X_part==k)
            
            if iter>0:
                if all((partsum[iter-1]-partsum[iter])==0) or iter==max_iter or abs(obj[-1]-obj[-2])<tol:
                    C_final.append(C)
                    obj_final.append(obj[-1])
                    part_final.append(X_part)             
                    break
            
            # "M-step" - update the cluster centers
            for k in range(K):
                idx_k = X_part==k
                V = np.reshape(np.swapaxes(X[idx_k],0,1),(p,np.sum(idx_k)*q))
                U,_,_ = scipy.sparse.linalg.svds(V,q)
                C[k] = U[:,:q]
            iter += 1
    best = np.nanargmax(np.array(obj_final))
    
    return C_final[best],part_final[best],obj_final[best]

def weighted_grassmannian_clustering(X,X_weights,K,max_iter=10000,tol=1e-16):
    """"
    Weighted grassmannian clustering using the chordal distance function and a SVD-based update rule
    
    X: size (nxpxq), where n is the number of observations, p is the number of features and q is the subspace dimensionality
    X_weights: size (n,q), where n is the number of observations and q is the subspace dimensionality (corresponds to eigenvalues)
    K: number of clusters
    max_iter: maximum number of iterations
    tol: tolerance for convergence

    """
    
    n,p,q = X.shape

    # initialize cluster centers using a normal distribution projected to the Grassmannian
    C = np.random.randn(K,p,q)
    C_weights = np.ones((K,q))
    for k in range(K):
        C[k] = C[k]@scipy.linalg.sqrtm(np.linalg.inv(C[k].T@C[k])) # project onto the Grassmannian

    # initialize counters
    iter = 0
    obj = []
    partsum = np.zeros((max_iter,K))
    while True:
        # "E-step" - compute the similarity between each matrix and each cluster center
        dis = 1/np.sqrt(2)(np.sum(X_weights**4)+np.sum(C_weights**4)-2*np.linalg.norm(np.swapaxes((X*X_weights[:,None,:])[:,None],-2,-1)@(C*C_weights[:,None,:])[None],axis=(-2,-1)))
        sim = -dis
        maxsim = np.max(sim,axis=1) # find the maximum similarity - the sum of this value is the objective function
        X_part = np.argmax(sim,axis=1) # assign each point to the cluster with the highest similarity
        obj.append(np.sum(maxsim))

        # check for convergence
        for k in range(K):
            partsum[iter,k] = np.sum(X_part==k)
        if iter>0:
            if all((partsum[iter-1]-partsum[iter])==0) or iter==max_iter or abs(obj[-1]-obj[-2])<tol:
                break
        
        # "M-step" - update the cluster centers
        for k in range(K):
            idx_k = X_part==k
            V = np.reshape(np.swapaxes(X[idx_k]*X_weights[idx_k,None,:],0,1),(p,np.sum(idx_k)*q))
            U,S,_ = scipy.sparse.linalg.svds(V,q)
            C[k] = U[:,:q]
            C_weights[k] = S**2

        iter += 1
    
    return C,C_weights,obj,X_part