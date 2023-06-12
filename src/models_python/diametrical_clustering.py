import numpy as np

def diametrical_clustering(X,K,max_iter=10000,num_repl=1,init=None,call=0):
    n,p = X.shape

    obj_final = []
    part_final = []
    C_final = []

    for _ in range(num_repl):
        if init is None or init=='++' or init=='plusplus' or init == 'diametrical_clustering_plusplus':
            C = diametrical_clustering_plusplus(X,K)
        else:
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
                if all((partsum[iter-1]-partsum[iter])==0) or iter==max_iter:
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
    
    return C

if __name__=='__main__':
    import matplotlib.pyplot as plt
    K = np.array(2)
    
    p = np.array(3)

    data = np.loadtxt('synth_data_2.csv',delimiter=',')
    data = data[np.arange(2000,step=2),:]

    C,part,obj = diametrical_clustering(X=data,K=K,num_repl=1,init=None)
    C = diametrical_clustering_plusplus(X=data,K=K)
    
    stop=7
