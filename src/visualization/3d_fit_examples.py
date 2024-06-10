# %%
#%%
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame()
num_repeats = 10
scale = 0.5
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

# %%
theta1 = np.array([0,np.pi/2,np.pi])
theta2 = np.array([0,np.pi,np.pi/2])

u_all1 = np.zeros((num_points_per_cluster,3,2))
u_all2 = np.zeros((num_points_per_cluster,3,2))
for n in range(num_points_per_cluster):
    theta_random = np.array([theta1[0]+np.random.random()*scale,theta1[1]+np.random.random()*scale,theta1[2]+np.random.random()*scale])
    coh_map = np.outer(np.cos(theta_random),np.cos(theta_random))+np.outer(np.sin(theta_random),np.sin(theta_random))
    l,u = np.linalg.eig(coh_map)
    order = np.argsort(l)[::-1]
    u_all1[n,:,0] = u[:,order[0]]
    u_all1[n,:,1] = u[:,order[1]]

    theta_random = np.array([theta2[0]+np.random.random()*scale,theta2[1]+np.random.random()*scale,theta2[2]+np.random.random()*scale])
    coh_map = np.outer(np.cos(theta_random),np.cos(theta_random))+np.outer(np.sin(theta_random),np.sin(theta_random))
    l,u = np.linalg.eig(coh_map)
    order = np.argsort(l)[::-1]
    u_all2[n,:,0] = u[:,order[0]]
    u_all2[n,:,1] = u[:,order[1]]
u_all = np.concatenate((u_all1,u_all2),axis=0)
true_labels = np.zeros((2,num_points_per_cluster*2))
true_labels[0,:num_points_per_cluster] = 1
true_labels[1,num_points_per_cluster:] = 1

# %% [markdown]
# K-means with sign-flip

# %%
u_in = u_all[:,:,0]
for n in range(num_points_per_cluster*2):
    if np.sum(u_in[n]>0)>1:
        u_in[n] = -u_in[n]

for i in range(num_repeats):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(u_in)
    labels = np.zeros((2,num_points_per_cluster*2))
    labels[0,kmeans.labels_==0] = 1
    labels[1,kmeans.labels_==1] = 1
    nmi = calc_NMI(true_labels,labels)
    df = pd.concat([df,pd.DataFrame({'method':'kmeans','repeat':i,'centroids':[kmeans.cluster_centers_],'nmi':nmi,'label':[labels]})])

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
from src.DMM_EM.riemannian_clustering import grassmannian_clustering_gruber2006
u_in = u_all
for i in range(num_repeats):
    C,part,obj = grassmannian_clustering_gruber2006(u_in,2)
    labels = np.zeros((2,num_points_per_cluster*2))
    labels[0,part==0] = 1
    labels[1,part==1] = 1
    nmi = calc_NMI(true_labels,labels)
    df = pd.concat([df,pd.DataFrame({'method':'grassmann','repeat':i,'centroids':[C],'nmi':nmi,'label':[labels]})])

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
    params,train_posterior,loglik_curve = train_model(data_train=u_in,K=2,options=options)
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
    params,train_posterior,loglik_curve = train_model(data_train=u_in,K=2,options=options)
    train_NMI = calc_NMI(true_labels,np.double(np.array(train_posterior)))
    # Lambda = params['M']@np.swapaxes(params['M'],-2,-1)+np.eye(3)
    Lambda=params['Lambda']
    df = pd.concat([df,pd.DataFrame({'method':'ACG','repeat':i,'centroids':[Lambda],'nmi':train_NMI,'label':[labels]})])

# %%
u_in = u_all
options['modelname'] = 'MACG'
options['threads'] = 8
options['rank'] = 'fullrank'
for i in range(num_repeats):
    params,train_posterior,loglik_curve = train_model(data_train=u_in,K=2,options=options)
    train_NMI = calc_NMI(true_labels,np.double(np.array(train_posterior)))
    # Sigma = params['M']@np.swapaxes(params['M'],-2,-1)+np.eye(3)
    Sigma = params['Lambda']
    df = pd.concat([df,pd.DataFrame({'method':'MACG','repeat':i,'centroids':[Sigma],'nmi':train_NMI,'label':[labels]})])

# %%
df.to_pickle('cluster_results.pkl')
# save also the data
np.save('cluster_data.npy',u_all)


