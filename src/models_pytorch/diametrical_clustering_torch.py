import torch

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
