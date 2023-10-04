import numpy as np
import scipy
import matplotlib.pyplot as plt

def weighted_grassmannian_clustering(X,K,X_weight=None,max_iter=10000,num_repl=1,init=None,call=0,tol=1e-16):
    """
    Weighted Grassmannian clustering algorithm for clustering data on the Grassmannian manifold (for subspaces).
    This algorithm includes the option of weighting the distance measure by an associated weight for each frame in the subspace.
    
    Input:
        X: data matrix (n,p,q) where p is the dimensionality and q is the number of frames
        K: number of clusters to be estimated
        X_weight: weight for each frame in the subspace. May be none or a nxq matrix
        max_iter: maximum number of iterations
        num_repl: number of repetitions
        init: initialization method. Options are '++' (or 'plusplus' or 'diametrical_clustering_plusplus'), 'uniform' (or 'unif')
        call: number of times the function has been called recursively
        tol: tolerance for convergence
    Output:
        C: cluster centers
        part: partition
        obj: objective function value
    """
    n,p,q = X.shape
    if X_weight is None:
        X_weight = np.reshape(np.tile(np.array((p*0.7,p*0.3)),n),(n,q))
    max_distance = p**2*np.pi/2**2 #to weight the distance measure to be within 0 and 1.

    obj_final = [] # objective function collector
    part_final = [] # partition collector
    C_final = [] # cluster center collector

    # loop over the number of repetitions
    for _ in range(num_repl):

        # initialize cluster centers
        C_weight = np.ones((K,q))*(p/q)

        C = np.random.uniform(size=(K,p,q))
        for k in range(K):
            C[k] = C[k]@scipy.linalg.sqrtm(np.linalg.inv(C[k].T@C[k])) # project onto the Grassmannian

        iter = 0
        obj = [] # objective function
        partsum = np.zeros((max_iter,K))
        while True:
            # "E-step" - compute the similarity between each matrix and each cluster center
            dis = np.zeros((n,K))
            S_all = np.zeros((K,n,q))
            # theta_all = np.zeros((K,n,q))
            # note this can surely be optimized!! but cba
            for k in range(K):
                for i in range(n): 
                    _,S,_ = np.linalg.svd(X[i].T@C[k],full_matrices=False)
                    S_all[k,i] = S  
                    theta = np.arccos(np.clip(S,-1,1)) # will only ever be between 0 and 1 plus numerical instability
                    # theta_all[k,i] = theta
                    dis[i,k] = np.sqrt(np.sum(C_weight[k]/X_weight[i]*theta**2))
            dis = dis/max_distance # normalize the distance measure to be between 0 and 1
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
                # flag mean
                V = np.reshape(np.swapaxes(X[idx_k],0,1),(p,np.sum(idx_k)*q))*np.reshape(X_weight[idx_k],np.sum(idx_k)*2)
                U,S2,_ = np.linalg.svd(V,full_matrices=False)
                C[k] = U[:,:q]
                C_weight[k] = S2[:q]**2/np.sum(S2[:q]**2)*p

            iter += 1
    try:
        best = np.nanargmax(np.array(obj_final))
    except: 
        if call>4:
            raise ValueError('Diametrical clustering ++ didn''t work for 5 re-calls of the function')
        print('Weighted Grassmannian clustering returned nan. Repeating')
        return diametrical_clustering(X,K,max_iter=max_iter,num_repl=num_repl,init=init,call=call+1)
    
    return C_final[best],part_final[best],obj_final[best]

def grassmannian_clustering_gruber2006(X,K,max_iter=10000,num_repl=1,init=None,call=0,tol=1e-16):
    
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
            # "E-step" - compute the similarity between each matrix and each cluster center
            dis = np.zeros((n,K))
            S_all = np.zeros((K,n,q))
            # note this can surely be optimized!! but cba
            # 

            for k in range(K):
                for i in range(n): 
                    dis[i,k] = 1/np.sqrt(2)*np.linalg.norm(X[i]@X[i].T-C[k]@C[k].T,'fro')
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
        return diametrical_clustering(X,K,max_iter=max_iter,num_repl=num_repl,init=init,call=call+1)
    
    return C_final[best],part_final[best],obj_final[best]


def diametrical_clustering(X,K,max_iter=10000,num_repl=1,init=None,call=0,tol=1e-16):
    """
    Diametrical clustering algorithm for clustering data on the sign-symmetric unit (hyper)sphere.
    Originally proposed in "Diametrical clustering for identifying
    anti-correlated gene clusters" by Dhillon IS et al., 2003 (Bioinformatics).
    Current version implemented from "The multivariate Watson distribution: 
    Maximum-likelihood estimation and other aspects" by Sra S & Karp D, 2012, Journal of Multivariate Analysis

    Input:
        X: data matrix (n,p)
        K: number of clusters
        max_iter: maximum number of iterations
        num_repl: number of repetitions
        init: initialization method. Options are '++' (or 'plusplus' or 'diametrical_clustering_plusplus'), 'uniform' (or 'unif')
        call: number of times the function has been called recursively
        tol: tolerance for convergence
    Output:
        C: cluster centers
        part: partition
        obj: objective function value
    """

    n,p = X.shape

    obj_final = [] # objective function collector
    part_final = [] # partition collector
    C_final = [] # cluster center collector

    # loop over the number of repetitions
    for _ in range(num_repl):
        if init is None or init=='++' or init=='plusplus' or init == 'diametrical_clustering_plusplus':
            C = diametrical_clustering_plusplus(X,K)
        elif init=='uniform' or init=='unif':
            C = np.random.uniform(size=(p,K))
            C = C/np.linalg.norm(C,axis=0)
        
        iter = 0
        obj = [] # objective function
        partsum = np.zeros((max_iter,K))
        while True:
            # "E-step" - compute the similarity between each point and each cluster center
            sim = (X@C)**2
            maxsim = np.max(sim,axis=1) # find the maximum similarity
            X_part = np.argmax(sim,axis=1) # assign each point to the cluster with the highest similarity
            obj.append(np.mean(maxsim))

            # check for convergence
            for k in range(K):
                partsum[iter,k] = np.sum(X_part==k)
            
            if iter>0:
                if all((partsum[iter-1]-partsum[iter])==0) or iter==max_iter or obj[-1]-obj[-2]<tol:
                    C_final.append(C)
                    obj_final.append(obj[-1])
                    part_final.append(X_part)             
                    break
            
            # "M-step" - update the cluster centers
            for k in range(K):
                idx_k = X_part==k
                A = X[idx_k].T@X[idx_k]
                C[:,k] = A@C[:,k]
                # C[:,k] = scipy.sparse.linalg.svds(A,k=1)[2][0] #gives the same result as above but
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
    """
    Diametrical clustering plusplus - initialization strategy for diametrical clustering.
    Input:
        X: data matrix (n,p)
        K: number of clusters
    Output:
        C: cluster centers


    """
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
    import matplotlib
    experiment = 1 #synthetic data experiment

    if experiment==0:
        K = np.array(2)
        p = np.array(3)

        # load synthetic dataset generated from the MACG model
        data = np.loadtxt('data/synthetic/synth_data_MACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
        n = data.shape[0]
        data_gr = np.zeros((int(n/2),p,2))
        data_gr[:,:,0] = data[np.arange(n,step=2),:] # first frame
        data_gr[:,:,1] = data[np.arange(n,step=2)+1,:] # second frame

        C_wgr,part,obj = weighted_grassmannian_clustering(X=data_gr,K=K,X_weight=None,num_repl=1,init=None)
        C_gr,part,obj = grassmannian_clustering_gruber2006(X=data_gr,K=K,num_repl=1,init=None)

        # load synthetic dataset generated from the ACG model with same ground truth as above
        data_sphere = np.loadtxt('data/synthetic/synth_data_ACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
        # data = data[np.arange(2000,step=2),:]

        C_dm,part,obj = diametrical_clustering(X=data_sphere,K=K,num_repl=1,init=None)
        
    elif experiment == 1:
        import h5py
        K=2
        num_subjects = 25
        data = np.array(h5py.File('data/processed/fMRI_SchaeferTian116_GSR_RL2.h5', 'r')['Dataset'][:,:num_subjects*1200*2]).T
        eigenvalues = np.array(h5py.File('data/processed/fMRI_SchaeferTian116_GSR_RL2.h5', 'r')['Eigenvalues'][:,:num_subjects*1200]).T
        p = data.shape[1]

        n = data.shape[0]
        data_gr = np.zeros((int(n/2),p,2))
        data_gr[:,:,0] = data[np.arange(n,step=2),:] # first frame
        data_gr[:,:,1] = data[np.arange(n,step=2)+1,:] # second frame
        data_sphere = data_gr[:,:,0]

        C_wgr,part,obj = weighted_grassmannian_clustering(X=data_gr,K=K,X_weight=eigenvalues,num_repl=1,init=None)
        C_gr,part,obj = grassmannian_clustering_gruber2006(X=data_gr,K=K,num_repl=1,init=None)

        C_dm,part,obj = diametrical_clustering(X=data_sphere,K=K,num_repl=1,init=None)

    # plot the "true" and estimated centroids in a 2,2 subplot
    import matplotlib
    matplotlib.rcParams.update({'font.size': 6})
    fig,ax = plt.subplots(3,4,figsize=(12,8))
    ax[0,0].barh(np.arange(p),C_dm[:,0])
    ax[0,0].set_title('Diametrical clustering centroid 1')
    ax[0,0].set_xlim([-1,1])
    # ax[0,1].barh(np.arange(p),data_gr[0,:,1])
    # ax[0,1].set_title('True centroid 1 (frame 2)')
    ax[0,1].set_xlim([-1,1])
    ax[0,2].barh(np.arange(p),C_dm[:,-1])
    ax[0,2].set_title('Diametrical clustering centroid 2')
    ax[0,2].set_xlim([-1,1])
    # ax[0,3].barh(np.arange(p),data_gr[-1,:,1])
    # ax[0,3].set_title('True centroid 2 (frame 2)')
    ax[0,3].set_xlim([-1,1])
    ax[1,0].barh(np.arange(p),C_wgr[0,:,0])
    ax[1,0].set_title('Weighted Grassmann centroid 1 (frame 1)')
    ax[1,0].set_xlim([-1,1])
    ax[1,1].barh(np.arange(p),C_wgr[0,:,1])
    ax[1,1].set_title('Weighted Grassmann centroid 1 (frame 2)')
    ax[1,1].set_xlim([-1,1])
    ax[1,2].barh(np.arange(p),C_wgr[1,:,0])
    ax[1,2].set_title('Weighted Grassmann centroid 2 (frame 1)')
    ax[1,2].set_xlim([-1,1])
    ax[1,3].barh(np.arange(p),C_wgr[1,:,1])
    ax[1,3].set_title('Weighted Grassmann centroid 2 (frame 2)')
    ax[1,3].set_xlim([-1,1])
    ax[2,0].barh(np.arange(p),C_gr[0,:,0])
    ax[2,0].set_title('Grassmann centroid 1 (frame 1)')
    ax[2,0].set_xlim([-1,1])
    ax[2,1].barh(np.arange(p),C_gr[0,:,1])
    ax[2,1].set_title('Grassmann centroid 1 (frame 2)')
    ax[2,1].set_xlim([-1,1])
    ax[2,2].barh(np.arange(p),C_gr[1,:,0])
    ax[2,2].set_title('Grassmann centroid 2 (frame 1)')
    ax[2,2].set_xlim([-1,1])
    ax[2,3].barh(np.arange(p),C_gr[1,:,1])
    ax[2,3].set_title('Grassmann centroid 2 (frame 2)')
    ax[2,3].set_xlim([-1,1])
    plt.show()

    plt.figure(),
    plt.subplot(3,2,1),plt.imshow(C_wgr[0]@C_wgr[0].T),plt.colorbar(),plt.title('Weighted Grassmann centroid 1')
    plt.subplot(3,2,2),plt.imshow(C_wgr[1]@C_wgr[1].T),plt.colorbar(),plt.title('Weighted Grassmann centroid 2')
    plt.subplot(3,2,3),plt.imshow(C_gr[0]@C_gr[0].T),plt.colorbar(),plt.title('Grassmann centroid 1')
    plt.subplot(3,2,4),plt.imshow(C_gr[1]@C_gr[1].T),plt.colorbar(),plt.title('Grassmann centroid 2')
    plt.subplot(3,2,5),plt.imshow(np.outer(C_dm[:,0],C_dm[:,0])),plt.colorbar(),plt.title('Diametrical clustering centroid 1')
    plt.subplot(3,2,6),plt.imshow(np.outer(C_dm[:,1],C_dm[:,1])),plt.colorbar(),plt.title('Diametrical clustering centroid 2')
    plt.show()
    
    stop=7
