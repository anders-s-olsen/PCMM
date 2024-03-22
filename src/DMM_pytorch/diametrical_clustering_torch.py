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
            C = diametrical_clustering_plusplus_torch(X,K)
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
                C[:,k] = A@C[:,k]
            C = C/torch.linalg.norm(C,axis=0)
            iter += 1
    # try:
    #     best = np.nanargmax(np.array(obj_final))
    # except: 
    #     if call>4:
    #         raise ValueError('Diametrical clustering ++ didn''t work for 5 re-calls of the function')
    #     print('Diametrical clustering returned nan. Repeating')
    #     return diametrical_clustering(X,K,max_iter=max_iter,num_repl=num_repl,init=init,call=call+1)
    best = torch.argmax(torch.tensor(obj_final))
    
    return C_final[best]#,part_final[best],obj_final[best]

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
    
    return C

def grassmannian_clustering_gruber2006_torch(X,K,max_iter=10000,num_repl=1,init=None,call=0,tol=1e-16):
    
    n,p,q = X.shape

    obj_final = [] # objective function collector
    part_final = [] # partition collector
    C_final = [] # cluster center collector
    XXt = X@torch.swapaxes(X,-2,-1)

    # loop over the number of repetitions
    for _ in range(num_repl):
        # initialize cluster centers
        C = torch.rand(K,p,q)
        for k in range(K):
            #eigenvals
            # evals = torch.linalg.eigvals(torch.linalg.inv(C[k].T@C[k]))
            # C[k] = C[k]@torch.sqrt(evals) # project onto the Grassmannian
            C[k] = C[k]@torch.tensor(scipy.linalg.sqrtm(torch.linalg.inv(C[k].T@C[k]).numpy()))

        iter = 0
        obj = [] # objective function
        partsum = torch.zeros(max_iter,K)
        while True:
            # "E-step" - compute the similarity between each matrix and each cluster center
            # dis = torch.zeros(n,K)
            # S_all = np.zeros((K,n,q))
            # note this can surely be optimized!! but cba
            # 
            CCt = C@torch.swapaxes(C,-2,-1)

            dis = 1/torch.sqrt(torch.tensor(2))*torch.linalg.matrix_norm(XXt[:,None]-CCt[None])
            sim = 1-dis
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
                V = torch.sum(X[idx_k]@torch.swapaxes(X[idx_k],1,2),dim=0)
                L,U = torch.linalg.eigh(V)
                # take only the last q columns
                # U = U[:,-q:]
                #rearrange the last two columns
                U = torch.index_select(U,1,torch.argsort(L,dim=0,descending=True))
                U = U[:,:q]
                # L,U = scipy.sparse.linalg.eigsh(V,k=q,which='LM')

                # U,S,_ = np.linalg.svd(np.reshape(np.swapaxes(X[idx_k],0,1),(p,np.sum(idx_k)*q)),full_matrices=False)
                C[k] = U
            iter += 1
    try:
        best = torch.argmax(torch.tensor(obj_final))
    except: 
        if call>4:
            raise ValueError('Diametrical clustering ++ didn''t work for 5 re-calls of the function')
        print('Weighted Grassmannian clustering returned nan. Repeating')
        return grassmannian_clustering_gruber2006_torch(X,K,max_iter=max_iter,num_repl=num_repl,init=init,call=call+1)
    
    return C_final[best]

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    K = torch.tensor(2)
    
    p = torch.tensor(3)

    data = torch.tensor(np.loadtxt('data/synthetic/synth_data_ACG.csv',delimiter=','))
    # data = data[np.arange(2000,step=2),:]

    C,part,obj = diametrical_clustering_torch(X=data,K=K,num_repl=1,init=None)
    # C = diametrical_clustering_plusplus_torch(X=data,K=K)
    
    stop=7
