# %%
#%%
import numpy as np
import h5py as h5
# from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2
# from kmeans import KMeans
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame()
num_repeats = 1
scale = 1
num_points_per_cluster = 1000

# %%
def calc_MI(Z1,Z2):
    P=Z1@Z2.T
    PXY=P/np.sum(P)
    PXPY=np.outer(np.sum(PXY,axis=1),np.sum(PXY,axis=0))
    ind=np.where(PXY>0)
    MI=np.sum(PXY[ind]*np.log(PXY[ind]/PXPY[ind]))
    return MI

def calc_NMI(Z1,Z2):
    Z1 = np.double(Z1)
    Z2 = np.double(Z2)
    #Z1 and Z2 are two partition matrices of size (KxN) where K is number of components and N is number of samples
    NMI = (2*calc_MI(Z1,Z2))/(calc_MI(Z1,Z1)+calc_MI(Z2,Z2))
    return NMI

def plot_coh_matrix2(coh_map1):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.imshow(coh_map1, vmin=-1,vmax=1, cmap='bwr')
    # add numbers in text for each cell
    for i in range(3):
        for j in range(3):
            ax.text(j,i, np.round(coh_map1[i,j],2), ha='center', va='center', color='k', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

# %%
theta1 = np.array([0,np.pi/2,np.pi])
theta2 = np.array([0,np.pi,np.pi/2])

u_all1 = np.zeros((num_points_per_cluster,3,2))
l_all1 = np.zeros((num_points_per_cluster,2))
u_all2 = np.zeros((num_points_per_cluster,3,2))
l_all2 = np.zeros((num_points_per_cluster,2))
for n in range(num_points_per_cluster):
    random_point = np.random.random(3)*scale-scale/2
    theta_random = theta1+random_point
    # theta_random = np.array([theta1[0]+np.random.random()*scale,theta1[1]+np.random.random()*scale,theta1[2]+np.random.random()*scale])
    coh_map = np.outer(np.cos(theta_random),np.cos(theta_random))+np.outer(np.sin(theta_random),np.sin(theta_random))
    l,u = np.linalg.eig(coh_map)
    order = np.argsort(l)[::-1]
    u_all1[n,:,0] = u[:,order[0]]
    l_all1[n,0] = l[order[0]]
    u_all1[n,:,1] = u[:,order[1]]
    l_all1[n,1] = l[order[1]]

    random_point = np.random.random(3)*scale-scale/2
    theta_random = theta2+random_point
    # theta_random = np.array([theta2[0]+np.random.random()*scale,theta2[1]+np.random.random()*scale,theta2[2]+np.random.random()*scale])
    coh_map = np.outer(np.cos(theta_random),np.cos(theta_random))+np.outer(np.sin(theta_random),np.sin(theta_random))
    l,u = np.linalg.eig(coh_map)
    order = np.argsort(l)[::-1]
    u_all2[n,:,0] = u[:,order[0]]
    l_all2[n,0] = l[order[0]]
    u_all2[n,:,1] = u[:,order[1]]
    l_all2[n,1] = l[order[1]]
u_all = np.concatenate((u_all1,u_all2),axis=0)
l_all = np.concatenate((l_all1,l_all2),axis=0)
true_labels = np.zeros((2,num_points_per_cluster*2))
true_labels[0,:num_points_per_cluster] = 1
true_labels[1,num_points_per_cluster:] = 1

# save also the data
np.save('src/visualization/fits/cluster_data.npy',u_all)
np.save('src/visualization/fits/cluster_evs.npy',l_all)
with h5.File('src/visualization/fits/cluster_data.h5', 'w') as f:
    f.create_dataset('data', data=u_all)
    f.create_dataset('evs', data=l_all)


coh_map1 = np.array(([1,0,-1],[0,1,0],[-1,0,1]))
ax = plot_coh_matrix2(coh_map1)
plt.savefig('src/visualization/figs/coherence_matrices_true1.png',bbox_inches='tight',dpi=300)

coh_map2 = np.array(([1,-1,0],[-1,1,0],[0,0,1]))
ax = plot_coh_matrix2(coh_map2)
plt.savefig('src/visualization/figs/coherence_matrices_true2.png',bbox_inches='tight',dpi=300)


# %% [markdown]
# K-means with sign-flip

# %%
u_in = u_all[:,:,0]
for n in range(num_points_per_cluster*2):
    if np.sum(u_in[n]>0)>1:
        u_in[n] = -u_in[n]

for i in range(num_repeats):
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(u_in)
    kmeans_cluster_centers,kmeans_labels = kmeans2(u_in,2,minit='++')
    labels = np.zeros((2,num_points_per_cluster*2))
    labels[0,kmeans_labels==0] = 1
    labels[1,kmeans_labels==1] = 1
    nmi = calc_NMI(true_labels,labels)
    df = pd.concat([df,pd.DataFrame({'method':'kmeans','repeat':i,'centroids':[kmeans_cluster_centers],'nmi':nmi,'label':[labels]})])

# %% [markdown]
# Diametrical clustering

# %%
from src.DMM_EM.riemannian_clustering import diametrical_clustering
u_in = u_all[:,:,0]
for i in range(num_repeats):
    C,part,obj = diametrical_clustering(u_in,2)
    labels = np.zeros((2,num_points_per_cluster*2))
    labels[0,part==0] = 1
    labels[1,part==1] = 1
    nmi = calc_NMI(true_labels,labels)
    df = pd.concat([df,pd.DataFrame({'method':'diametrical','repeat':i,'centroids':[C],'nmi':nmi,'label':[labels]})])

# %% [markdown]
# Grassmann clustering

# %%
from src.DMM_EM.riemannian_clustering import grassmann_clustering
u_in = u_all
for i in range(num_repeats):
    C,part,obj = grassmann_clustering(u_in,2)
    labels = np.zeros((2,num_points_per_cluster*2))
    labels[0,part==0] = 1
    labels[1,part==1] = 1
    nmi = calc_NMI(true_labels,labels)
    df = pd.concat([df,pd.DataFrame({'method':'grassmann','repeat':i,'centroids':[C],'nmi':nmi,'label':[labels]})])

coh_map1 = C[0]@C[0].T
ax = plot_coh_matrix2(coh_map1)
plt.savefig('src/visualization/figs/coherence_matrices_Gr1.png',bbox_inches='tight',dpi=300)

coh_map2 = C[1]@C[1].T
ax = plot_coh_matrix2(coh_map2)
plt.savefig('src/visualization/figs/coherence_matrices_Gr2.png',bbox_inches='tight',dpi=300)



# %%
from src.DMM_EM.riemannian_clustering import weighted_grassmann_clustering
u_in = u_all
l_in = l_all
for i in range(num_repeats):
    C,C_weights,part,obj = weighted_grassmann_clustering(u_in,K=2,X_weights=l_in)
    labels = np.zeros((2,num_points_per_cluster*2))
    labels[0,part==0] = 1
    labels[1,part==1] = 1
    nmi = calc_NMI(true_labels,labels)
    df = pd.concat([df,pd.DataFrame({'method':'weighted_grassmann','repeat':i,'centroids':[C],'centroids_weights':[C_weights],'nmi':nmi,'label':[labels]})])

# %% [markdown]
# mixture model setup

# %%
from src.helper_functions import train_model
options = {}
options['init'] = 'dc_seg'
options['LR'] = 0
options['tol'] = 1e-8
options['max_iter'] = 1000
options['num_repl_inner'] = 1
options['threads'] = 8
options['HMM'] = False
if options['LR']!=0:
    import torch
    u_all = torch.tensor(u_all)

# %% [markdown]
# Watson mixture models

# %%
u_in = u_all[:,:,0]
options['modelname'] = 'Watson'
options['threads'] = 8
options['rank'] = 1
for i in range(num_repeats):
    params,train_posterior,loglik_curve = train_model(data_train=u_in,L_train=None,K=2,options=options)
    train_NMI = calc_NMI(true_labels,np.double(np.array(train_posterior)))
    df = pd.concat([df,pd.DataFrame({'method':'Watson','repeat':i,'centroids':[params],'nmi':train_NMI,'label':[labels]})])

# %% [markdown]
# ACG mixture models

# %%
u_in = u_all[:,:,0]
options['modelname'] = 'ACG'
options['threads'] = 8
options['rank'] = 'fullrank'
for i in range(num_repeats):
    params,train_posterior,loglik_curve = train_model(data_train=u_in,L_train=None,K=2,options=options)
    train_NMI = calc_NMI(true_labels,np.double(np.array(train_posterior)))
    # Lambda = params['M']@np.swapaxes(params['M'],-2,-1)+np.eye(3)
    Lambda=params['Lambda']
    df = pd.concat([df,pd.DataFrame({'method':'ACG_fullrank','repeat':i,'centroids':[Lambda],'nmi':train_NMI,'label':[labels]})])

coh_map1 = Lambda[0]
ax = plot_coh_matrix2(coh_map1)
plt.savefig('src/visualization/figs/coherence_matrices_ACG1.png',bbox_inches='tight',dpi=300)

coh_map2 = Lambda[1]
ax = plot_coh_matrix2(coh_map2)
plt.savefig('src/visualization/figs/coherence_matrices_ACG2.png',bbox_inches='tight',dpi=300)


# %%
# u_in = u_all[:,:,0]
# options['modelname'] = 'ACG'
# options['threads'] = 8
# options['rank'] = 3
# for i in range(num_repeats):
#     params,train_posterior,loglik_curve = train_model(data_train=u_in,L_train=None,K=2,options=options)
#     train_NMI = calc_NMI(true_labels,np.double(np.array(train_posterior)))
#     Lambda = params['M']@np.swapaxes(params['M'],-2,-1)+np.eye(3)
#     # Lambda=params['Lambda']
#     df = pd.concat([df,pd.DataFrame({'method':'ACG_lowrank','repeat':i,'centroids':[Lambda],'nmi':train_NMI,'label':[labels]})])

# %%
u_in = u_all
options['modelname'] = 'MACG'
options['threads'] = 8
options['rank'] = 'fullrank'
for i in range(num_repeats):
    params,train_posterior,loglik_curve = train_model(data_train=u_in,L_train=None,K=2,options=options)
    train_NMI = calc_NMI(true_labels,np.double(np.array(train_posterior)))
    # Sigma = params['M']@np.swapaxes(params['M'],-2,-1)+np.eye(3)
    Sigma = params['Lambda']
    df = pd.concat([df,pd.DataFrame({'method':'MACG_fullrank','repeat':i,'centroids':[Sigma],'nmi':train_NMI,'label':[labels]})])

coh_map1 = Sigma[0]
ax = plot_coh_matrix2(coh_map1)
plt.savefig('src/visualization/figs/coherence_matrices_ACG1.png',bbox_inches='tight',dpi=300)

coh_map2 = Sigma[1]
ax = plot_coh_matrix2(coh_map2)
plt.savefig('src/visualization/figs/coherence_matrices_ACG2.png',bbox_inches='tight',dpi=300)


# %%
# u_in = u_all
# options['modelname'] = 'MACG'
# options['threads'] = 8
# options['rank'] = 3
# for i in range(num_repeats):
#     params,train_posterior,loglik_curve = train_model(data_train=u_in,L_train=None,K=2,options=options)
#     train_NMI = calc_NMI(true_labels,np.double(np.array(train_posterior)))
#     Sigma = params['M']@np.swapaxes(params['M'],-2,-1)+np.eye(3)
#     # Sigma = params['Lambda']
#     df = pd.concat([df,pd.DataFrame({'method':'MACG_lowrank','repeat':i,'centroids':[Sigma],'nmi':train_NMI,'label':[labels]})])

# %%
u_in = u_all
l_in = l_all
options['modelname'] = 'SingularWishart'
options['threads'] = 8
options['rank'] = 'fullrank'
for i in range(num_repeats):
    params,train_posterior,loglik_curve = train_model(data_train=u_in,L_train=l_in,K=2,options=options)
    train_NMI = calc_NMI(true_labels,np.double(np.array(train_posterior)))
    # Psi = params['M']@np.swapaxes(params['M'],-2,-1)+np.eye(3)
    Psi = params['Lambda']
    df = pd.concat([df,pd.DataFrame({'method':'SingularWishart_fullrank','repeat':i,'centroids':[Psi],'nmi':train_NMI,'label':[labels]})])

# %%
# u_in = u_all
# l_in = l_all
# options['modelname'] = 'SingularWishart'
# options['threads'] = 8
# options['rank'] = 3
# for i in range(num_repeats):
#     params,train_posterior,loglik_curve = train_model(data_train=u_in,L_train=l_in,K=2,options=options)
#     train_NMI = calc_NMI(true_labels,np.double(np.array(train_posterior)))
#     Psi = params['M']@np.swapaxes(params['M'],-2,-1)+np.eye(3)
#     # Psi = params['Lambda']
#     df = pd.concat([df,pd.DataFrame({'method':'SingularWishart_lowrank','repeat':i,'centroids':[Psi],'nmi':train_NMI,'label':[labels]})])

# # %%
# df

# %%
df.to_pickle('src/visualization/fits/cluster_results.pkl')
np.savetxt('src/visualization/fits/kmeans_centroids.txt',df[df['method']=='kmeans'].centroids.iloc[0])
np.savetxt('src/visualization/fits/kmeans_labels.txt',df[df['method']=='kmeans'].label.iloc[0])
np.savetxt('src/visualization/fits/diametrical_centroids.txt',df[df['method']=='diametrical'].centroids.iloc[0])
np.savetxt('src/visualization/fits/diametrical_labels.txt',df[df['method']=='diametrical'].label.iloc[0])
np.savetxt('src/visualization/fits/grassmann_centroids1.txt',df[df['method']=='grassmann'].centroids.iloc[0][0])
np.savetxt('src/visualization/fits/grassmann_centroids2.txt',df[df['method']=='grassmann'].centroids.iloc[0][1])
np.savetxt('src/visualization/fits/grassmann_labels.txt',df[df['method']=='grassmann'].label.iloc[0])
np.savetxt('src/visualization/fits/weighted_grassmann_centroids1.txt',df[df['method']=='weighted_grassmann'].centroids.iloc[0][0])
np.savetxt('src/visualization/fits/weighted_grassmann_centroids_weights1.txt',df[df['method']=='weighted_grassmann'].centroids_weights.iloc[0][0])
np.savetxt('src/visualization/fits/weighted_grassmann_centroids_weights2.txt',df[df['method']=='weighted_grassmann'].centroids_weights.iloc[0][1])
np.savetxt('src/visualization/fits/weighted_grassmann_centroids2.txt',df[df['method']=='weighted_grassmann'].centroids.iloc[0][1])
np.savetxt('src/visualization/fits/weighted_grassmann_labels.txt',df[df['method']=='weighted_grassmann'].label.iloc[0])
np.savetxt('src/visualization/fits/watson_centroids_mu.txt',df[df['method']=='Watson'].centroids.iloc[0]['mu'])
np.savetxt('src/visualization/fits/watson_centroids_kappa.txt',df[df['method']=='Watson'].centroids.iloc[0]['kappa'])
np.savetxt('src/visualization/fits/watson_centroids_pi.txt',df[df['method']=='Watson'].centroids.iloc[0]['pi'])
np.savetxt('src/visualization/fits/watson_labels.txt',df[df['method']=='Watson'].label.iloc[0])
np.savetxt('src/visualization/fits/ACG_centroids1.txt',df[df['method']=='ACG_fullrank'].centroids.iloc[0][0])
np.savetxt('src/visualization/fits/ACG_centroids2.txt',df[df['method']=='ACG_fullrank'].centroids.iloc[0][1])
np.savetxt('src/visualization/fits/ACG_labels.txt',df[df['method']=='ACG_fullrank'].label.iloc[0])
np.savetxt('src/visualization/fits/MACG_centroids1.txt',df[df['method']=='MACG_fullrank'].centroids.iloc[0][0])
np.savetxt('src/visualization/fits/MACG_centroids2.txt',df[df['method']=='MACG_fullrank'].centroids.iloc[0][1])
np.savetxt('src/visualization/fits/MACG_labels.txt',df[df['method']=='MACG_fullrank'].label.iloc[0])
np.savetxt('src/visualization/fits/SingularWishart_centroids1.txt',df[df['method']=='SingularWishart_fullrank'].centroids.iloc[0][0])
np.savetxt('src/visualization/fits/SingularWishart_centroids2.txt',df[df['method']=='SingularWishart_fullrank'].centroids.iloc[0][1])
np.savetxt('src/visualization/fits/SingularWishart_labels.txt',df[df['method']=='SingularWishart_fullrank'].label.iloc[0])
