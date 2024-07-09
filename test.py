import numpy as np
from src.DMM_EM.riemannian_clustering import weighted_grassmann_clustering

def coh_map(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    coh_map = np.outer(c,c)+np.outer(s,s)
    return coh_map

def generate_data(p, K, N, noise_scale=0.2):
    noise = np.random.rand(p,N) * noise_scale
    data_gr = np.empty((N,p,2))
    data_eigvals = np.empty((N,2))
    theta_bases = [] # cluster_base_vectors
 
    for i in range(K):
        theta_base = np.ones(p)*np.pi/2
        theta_base[i] = 0
        theta_bases.append(theta_base)
 
    for j in range(K):
        theta_cluster = np.asarray([theta_bases[j] for _ in range(N)]) + noise.T
        for i in range(N//K):
            eigvals,eigvecs = np.linalg.eigh(coh_map(theta_cluster[i+j*N//K]))
            eigvals,eigvecs = eigvals[-2:],eigvecs[:,-2:]
            data_gr[i+j*N//K,:,0] = eigvecs[:,np.argmax(eigvals)]
            data_gr[i+j*N//K,:,1] = eigvecs[:,np.argmin(eigvals)]
            data_eigvals[i] = eigvals
    data_sphere = data_gr[:,:,0]
    return data_sphere, data_gr, data_eigvals

if __name__ == '__main__':
    p = 10
    K = 2
    N = 100
    data_sphere, data_gr, data_eigvals = generate_data(p, K, N)
    out = weighted_grassmann_clustering(data_gr,data_eigvals,K)
    done=7