import numpy as np
import scipy

def diametrical_clustering(X,K,max_iter=10000,num_repl=1,init=None,call=0,tol=1e-16):
    n,p = X.shape

    obj_final = []
    part_final = []
    C_final = []

    for _ in range(num_repl):
        if init is None or init=='++' or init=='plusplus' or init == 'diametrical_clustering_plusplus':
            C = diametrical_clustering_plusplus(X,K)
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
    try:
        best = np.nanargmax(np.array(obj_final))
    except: 
        if call>4:
            raise ValueError('Diametrical clustering ++ didn''t work for 5 re-calls of the function')
        print('Diametrical clustering returned nan. Repeating')
        return diametrical_clustering(X,K,max_iter=max_iter,num_repl=num_repl,init=init,call=call+1)
    
    return C_final[best]

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
    
    return C

def grassmannian_clustering_gruber2006(X,K,max_iter=10000,num_repl=1,init=None,call=0,tol=1e-16):
    
    n,p,q = X.shape

    obj_final = [] # objective function collector
    part_final = [] # partition collector
    C_final = [] # cluster center collector
    XXt = X@np.swapaxes(X,-2,-1)

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
            # "E-step" - compute the similarity between each matrix and each cluster center
            dis = np.zeros((n,K))
            # S_all = np.zeros((K,n,q))
            # note this can surely be optimized!! but cba
            CCt = C@np.swapaxes(C,-2,-1)

            dis = 1/np.sqrt(2)*np.linalg.norm(XXt[:,None]-CCt[None],axis=(-2,-1))
            sim = 1-dis
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
                V = np.sum(X[idx_k]@np.swapaxes(X[idx_k],1,2),axis=0)
                L,U = scipy.sparse.linalg.eigsh(V,k=q,which='LM')

                # U,S,_ = np.linalg.svd(np.reshape(np.swapaxes(X[idx_k],0,1),(p,np.sum(idx_k)*q)),full_matrices=False)
                C[k] = U
            iter += 1
    try:
        best = np.nanargmax(np.array(obj_final))
    except: 
        if call>4:
            raise ValueError('Diametrical clustering ++ didn''t work for 5 re-calls of the function')
        print('Weighted Grassmannian clustering returned nan. Repeating')
        return grassmannian_clustering_gruber2006(X,K,max_iter=max_iter,num_repl=num_repl,init=init,call=call+1)
    
    return C_final[best]

if __name__=='__main__':
    import matplotlib.pyplot as plt
    K = np.array(2)
    
    p = np.array(3)

    data = np.loadtxt('synth_data_2.csv',delimiter=',')
    data = data[np.arange(2000,step=2),:]

    C,part,obj = diametrical_clustering(X=data,K=K,num_repl=1,init=None)
    C = diametrical_clustering_plusplus(X=data,K=K)
    
    stop=7
