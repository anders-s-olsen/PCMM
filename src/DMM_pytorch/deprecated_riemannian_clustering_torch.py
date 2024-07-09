import torch
torch.set_default_dtype(torch.float64)
import scipy

def diametrical_clustering_torch(X,K,max_iter=10000,num_repl=1,init=None,tol=1e-16):
    n,p = X.shape

    obj_final = []
    part_final = []
    C_final = []

    for _ in range(num_repl):
        if init is None or init=='++' or init=='plusplus' or init == 'diametrical_clustering_plusplus':
            C,_,_ = diametrical_clustering_plusplus_torch(X,K)
        else:
            C = torch.rand(size=(p,K))
            C = C/torch.linalg.norm(C,axis=0)
        
        iter = 0
        obj = []
        partsum = torch.zeros((max_iter,K))
        while True:
            # E-step
            dis = (X@C)**2
            maxdis = torch.max(dis,dim=1)[0] # check that this works
            X_part = torch.argmax(dis,dim=1)
            obj.append(torch.mean(maxdis))

            for k in range(K):
                partsum[iter,k] = torch.sum(X_part==k)
            
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
                C[:,k] = (C[:,k]) @ A
            C = C/torch.linalg.norm(C,axis=0)
            iter += 1
    
    best = torch.argmax(torch.tensor(obj_final))
    return C_final[best],part_final[best],obj_final[best]

def diametrical_clustering_plusplus_torch(X,K):
    n,_ = X.shape

    # choose first centroid at random from X
    idx = torch.multinomial(torch.ones(1000),num_samples=1).item()
    C = X[idx].clone()[:,None]
 
    # for all other centroids, compute the distance from all X to the current set of centroids. 
    # Construct a weighted probability distribution and sample using this. 

    for k in range(K-1):
        if idx!=0 and idx !=n-1:
            X = torch.vstack([X[:idx],X[idx+1:]])
        elif idx==0:
            X = X[1:]
        elif idx==n-1:
            X = X[:-1]
        dist = 1-(X@C)**2 #large means far away
        min_dist = torch.min(dist,dim=1)[0] #choose the distance to the closest centroid for each point
        prob_dist = min_dist/torch.sum(min_dist) # construct the prob. distribution
        # idx = np.random.choice(n-k-1,p=prob_dist)
        idx = torch.multinomial(prob_dist,num_samples=1).item()
        C = torch.hstack((C,X[idx].clone()[:,None]))
    
    dis = (X@C)**2
    maxdis = torch.max(dis,dim=1)[0] # check that this works
    X_part = torch.argmax(dis,dim=1)
    obj = torch.mean(maxdis)

    return C,X_part,obj

def grassmann_clustering_torch(X,K,max_iter=10000,num_repl=1,init=None,call=0,tol=1e-16):
    
    n,p,q = X.shape

    obj_final = [] # objective function collector
    part_final = [] # partition collector
    C_final = [] # cluster center collector

    # loop over the number of repetitions
    for _ in range(num_repl):
        # initialize cluster centers
        C = torch.rand(K,p,q)
        for k in range(K):
            C[k] = C[k]@torch.tensor(scipy.linalg.sqrtm(torch.linalg.inv(C[k].T@C[k]).numpy()))

        iter = 0
        obj = [] # objective function
        partsum = torch.zeros(max_iter,K)
        while True:
            # "E-step" - compute the similarity between each matrix and each cluster center
            
            dis = 1/torch.sqrt(2)*(2*q-2*torch.linalg.norm(torch.swapaxes(X[:,None],-2,-1)@C[None],axis=(-2,-1)))
            sim = -dis
            maxsim,X_part = torch.max(sim,dim=1) # find the maximum similarity
            obj.append(torch.mean(maxsim))

            # check for convergence
            for k in range(K):
                partsum[iter,k] = torch.sum(X_part==k)
            
            if iter>0:
                if all((partsum[iter-1]-partsum[iter])==0) or iter==max_iter or abs(obj[-1]-obj[-2])<tol:
                    C_final.append(C)
                    obj_final.append(obj[-1])
                    part_final.append(X_part)             
                    break
            
            # "M-step" - update the cluster centers
            for k in range(K):
                idx_k = X_part==k
                V = torch.reshape(torch.swapaxes(X[idx_k],0,1),(p,torch.sum(idx_k)*q))
                U,_,_ = torch.svd_lowrank(V,q=q)
                C[k] = U[:,:q]
            iter += 1
    best = torch.argmax(torch.tensor(obj_final))
    
    return C_final[best],part_final[best],obj_final[best]

def weighted_grassmann_clustering_torch(X,X_weights,K,max_iter=10000,tol=1e-16):
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
    C = torch.rand(K,p,q)
    C_weights = torch.ones(K,q)
    for k in range(K):
        C[k] = C[k]@scipy.linalg.sqrtm(torch.linalg.inv(C[k].T@C[k])) # project onto the Grassmannian

    # initialize counters
    iter = 0
    obj = []
    partsum = torch.zeros((max_iter,K))
    while True:
        # "E-step" - compute the similarity between each matrix and each cluster center
        dis = 1/torch.sqrt(2)*(torch.sum(X_weights**4)+torch.sum(C_weights**4)-2*torch.linalg.norm(torch.swapaxes((X*X_weights[:,None,:])[:,None],-2,-1)@(C*C_weights[:,None,:])[None],dim=(-2,-1)))
        sim = -dis
        maxsim,X_part = torch.max(sim,dim=1) # find the maximum similarity - the sum of this value is the objective function
        obj.append(torch.sum(maxsim))

        # check for convergence
        for k in range(K):
            partsum[iter,k] = torch.sum(X_part==k)
        if iter>0:
            if all((partsum[iter-1]-partsum[iter])==0) or iter==max_iter or abs(obj[-1]-obj[-2])<tol:
                break
        
        # "M-step" - update the cluster centers
        for k in range(K):
            idx_k = X_part==k
            V = torch.reshape(torch.swapaxes(X[idx_k]*X_weights[idx_k,None,:],0,1),(p,torch.sum(idx_k)*q))
            U,S,_ = torch.svd_lowrank(V,q=q)
            C[k] = U[:,:q]
            C_weights[k] = S**2

        iter += 1
    
    return C,obj,X_part