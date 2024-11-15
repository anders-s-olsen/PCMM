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
num_repeats = 10
scale = 0.1
scale2 = 3*np.pi/4
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
# theta1 = np.array([0,0,np.pi])
# theta2 = np.array([0,np.pi,0])

possible_phases = np.linspace(0,2*np.pi,1000)

u_all1 = np.zeros((num_points_per_cluster,3,2))
u_all2 = np.zeros((num_points_per_cluster,3,2))

n = num_points_per_cluster
for traintest in range(2):
    # for n in range(num_points_per_cluster):
    noises = np.zeros((3,n))
    noises[0] = np.random.uniform(0, scale, n)
    noises[1] = np.random.uniform(0, scale, n)
    noises[2] = np.random.uniform(0, scale, n)
    t = np.linspace(0, 5, n)
    signal6 = np.zeros((3,1000))
    signal6[0] = np.sin(2*np.pi*t)
    signal6[1] = np.sin(2*np.pi*t)
    signal6[2] = np.random.uniform(-scale2/2, scale2/2, n)
    signal7 = np.zeros((3,1000))
    signal7[0] = np.sin(2*np.pi*t)
    signal7[1] = np.random.uniform(-scale2/2, scale2/2, n)
    signal7[2] = np.sin(2*np.pi*t)
    u_all1[:,:,traintest] = ((signal6+noises)/np.linalg.norm(signal6+noises,axis=0)).T
    u_all2[:,:,traintest] = ((signal7+noises)/np.linalg.norm(signal7+noises,axis=0)).T
        # #draw a phase from the possible phases
        # theta1 = np.random.choice(possible_phases,1).item()
        # random_point = np.random.random(2)*scale-scale/2
        # random_random = np.random.random(1)*scale2-scale2/2
        # theta_random = np.array([theta1+random_point[0],random_random[0],theta1+random_point[1]])
        # u_all1[n,:,traintest] = np.cos(theta_random)/np.linalg.norm(np.cos(theta_random))

        # theta2 = np.random.choice(possible_phases,1).item()
        # random_point = np.random.random(2)*scale-scale/2
        # random_random = np.random.random(1)*scale2-scale2/2
        # theta_random = np.array([theta2+random_point[0],theta2+random_point[1],random_random[0]])
        # u_all2[n,:,traintest] = np.cos(theta_random)/np.linalg.norm(np.cos(theta_random))
        
u_all_train = np.concatenate((u_all1[:,:,0],u_all2[:,:,0]),axis=0)
u_all_test = np.concatenate((u_all1[:,:,1],u_all2[:,:,1]),axis=0)
true_labels = np.zeros((2,num_points_per_cluster*2))
true_labels[0,:num_points_per_cluster] = 1
true_labels[1,num_points_per_cluster:] = 1

# save also the data
np.save('src/visualization/fits/cluster_data.npy',u_all_train)
# np.save('src/visualization/fits/cluster_evs.npy',l_all)
with h5.File('src/visualization/fits/cluster_data.h5', 'w') as f:
    f.create_dataset('data', data=u_all_train)
    # f.create_dataset('evs', data=l_all)


from src.helper_functions import train_model,test_model
options = {}
options['init'] = 'dc'
options['LR'] = 0
options['tol'] = 1e-8
options['max_iter'] = 1000
options['num_repl_inner'] = 1
options['threads'] = 8
options['HMM'] = False
if options['LR']!=0:
    import torch
    u_all = torch.tensor(u_all_train)

# %% [markdown]
# Watson mixture models

# %%
u_in = u_all_train
u_in_test = u_all_test
df = pd.DataFrame()
for inner in range(num_repeats):
    print(inner)
    options['modelname'] = 'Watson'
    options['threads'] = 8
    options['rank'] = 1
    for K in range(1,11):
        params,train_posterior,loglik_curve = train_model(data_train=u_in,K=K,options=options)
        test_loglik,test_posterior,_ = test_model(data_test=u_in_test,params=params,K=K,options=options)
        np.savetxt('src/visualization/fits/watson_centroids_mu_K='+str(K)+'.txt',params['mu'])
        np.savetxt('src/visualization/fits/watson_centroids_kappa_K='+str(K)+'.txt',params['kappa'])
        np.savetxt('src/visualization/fits/watson_centroids_pi_K='+str(K)+'.txt',params['pi'])
        # np.savetxt('src/visualization/fits/watson_labels_K='+str(K)+'.txt',df[df['method']=='Watson'].label.iloc[0])
        df = pd.concat([df,pd.DataFrame([{'modelname':options['modelname'],'K':K,'train_loglik':loglik_curve[-1],'test_loglik':test_loglik,'inner':inner}])],ignore_index=True)

    options['modelname'] = 'ACG'
    options['threads'] = 8
    options['rank'] = 'fullrank'
    # options['rank'] = 2
    for K in range(1,11):
        params,train_posterior,loglik_curve = train_model(data_train=u_in,K=K,options=options)
        test_loglik,test_posterior,_ = test_model(data_test=u_in_test,params=params,K=K,options=options)
        for k in range(K):
            # Lambda_k = params['M'][k]@params['M'][k].T+np.eye(3)
            # Lambda_k = Lambda_k/np.trace(Lambda_k)
            # np.savetxt('src/visualization/fits/ACG_centroids_K='+str(K)+str(k)+'.txt',Lambda_k)
            np.savetxt('src/visualization/fits/ACG_centroids_K='+str(K)+str(k)+'.txt',params['Lambda'][k])
        np.savetxt('src/visualization/fits/ACG_centroids_pi_K='+str(K)+'.txt',params['pi'])
        # np.savetxt('src/visualization/fits/ACG_labels_K='+str(K)+'.txt',df[df['method']=='ACG_fullrank'].label.iloc[0])
        df = pd.concat([df,pd.DataFrame([{'modelname':options['modelname'],'K':K,'train_loglik':loglik_curve[-1],'test_loglik':test_loglik,'inner':inner}])],ignore_index=True)

    df.to_pickle('src/visualization/fits/cluster_results_rank.pkl')