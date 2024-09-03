import numpy as np
import h5py as h5
from scipy.cluster.vq import kmeans2
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame()
num_repeats = 25
num_points_per_cluster = 1000

# make noise levels from 0 to 2pi in 8 steps
noise_levels = np.linspace(np.pi/4,2*np.pi,8)

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

#make a list of pi/4, pi/2, 3*pi/4 etc
levels = np.linspace(np.pi/16,np.pi,16)


for scale in range(len(levels)):
    print('Working on '+str(levels[scale]))
    noise_scale = levels[scale]

    theta1 = np.array([0,np.pi/2,np.pi])
    theta2 = np.array([0,np.pi,np.pi/2])

    u_all1 = np.zeros((num_points_per_cluster,3,2))
    l_all1 = np.zeros((num_points_per_cluster,2))
    u_all2 = np.zeros((num_points_per_cluster,3,2))
    l_all2 = np.zeros((num_points_per_cluster,2))
    u_all_complex1 = np.zeros((num_points_per_cluster,3),dtype=complex)
    u_all_complex2 = np.zeros((num_points_per_cluster,3),dtype=complex)
    for n in range(num_points_per_cluster):
        random_point = np.random.random(3)*noise_scale-noise_scale/2
        theta_random = theta1+random_point
        # theta_random = np.array([theta1[0]+np.random.random()*scale,theta1[1]+np.random.random()*scale,theta1[2]+np.random.random()*scale])
        coh_map = np.outer(np.cos(theta_random),np.cos(theta_random))+np.outer(np.sin(theta_random),np.sin(theta_random))
        l,u = np.linalg.eig(coh_map)
        order = np.argsort(l)[::-1]
        u_all1[n,:,0] = u[:,order[0]]
        l_all1[n,0] = l[order[0]]
        u_all1[n,:,1] = u[:,order[1]]
        l_all1[n,1] = l[order[1]]

        coh_map_complex = np.outer(np.exp(1j*theta_random),np.exp(-1j*theta_random))
        l,u = np.linalg.eig(coh_map_complex)
        u_all_complex1[n,:] = u[:,np.argmax(l)]

        random_point = np.random.random(3)*noise_scale-noise_scale/2
        theta_random = theta2+random_point
        # theta_random = np.array([theta2[0]+np.random.random()*scale,theta2[1]+np.random.random()*scale,theta2[2]+np.random.random()*scale])
        coh_map = np.outer(np.cos(theta_random),np.cos(theta_random))+np.outer(np.sin(theta_random),np.sin(theta_random))
        l,u = np.linalg.eig(coh_map)
        order = np.argsort(l)[::-1]
        u_all2[n,:,0] = u[:,order[0]]
        l_all2[n,0] = l[order[0]]
        u_all2[n,:,1] = u[:,order[1]]
        l_all2[n,1] = l[order[1]]

        coh_map_complex = np.outer(np.exp(1j*theta_random),np.exp(-1j*theta_random))
        l,u = np.linalg.eig(coh_map_complex)
        u_all_complex2[n,:] = u[:,np.argmax(l)]

    u_all = np.concatenate((u_all1,u_all2),axis=0)
    l_all = np.concatenate((l_all1,l_all2),axis=0)
    u_all_complex = np.concatenate((u_all_complex1,u_all_complex2),axis=0)
    true_labels = np.zeros((2,num_points_per_cluster*2))
    true_labels[0,:num_points_per_cluster] = 1
    true_labels[1,num_points_per_cluster:] = 1

    ######################################################3
    # Kmeans
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
        df = pd.concat([df,pd.DataFrame({'method':'kmeans','repeat':i,'centroids':[kmeans_cluster_centers],'nmi':nmi,'label':[labels],'noise':scale})])

    ######################################################
    # diametrical clustering (real)
    from src.DMM_EM.riemannian_clustering import diametrical_clustering
    u_in = u_all[:,:,0]
    for i in range(num_repeats):
        C,part,obj = diametrical_clustering(u_in,2)
        labels = np.zeros((2,num_points_per_cluster*2))
        labels[0,part==0] = 1
        labels[1,part==1] = 1
        nmi = calc_NMI(true_labels,labels)
        df = pd.concat([df,pd.DataFrame({'method':'diametrical','repeat':i,'centroids':[C],'nmi':nmi,'label':[labels],'noise':scale})])

    ######################################################
    # diametrical clustering (complex)
    u_in = u_all_complex
    for i in range(num_repeats):
        C,part,obj = diametrical_clustering(u_in,2)
        labels = np.zeros((2,num_points_per_cluster*2))
        labels[0,part==0] = 1
        labels[1,part==1] = 1
        nmi = calc_NMI(true_labels,labels)
        df = pd.concat([df,pd.DataFrame({'method':'diametrical_complex','repeat':i,'centroids':[C],'nmi':nmi,'label':[labels],'noise':scale})])

    ######################################################
    # grassmann clustering
    from src.DMM_EM.riemannian_clustering import grassmann_clustering
    u_in = u_all
    for i in range(num_repeats):
        C,part,obj = grassmann_clustering(u_in,2)
        labels = np.zeros((2,num_points_per_cluster*2))
        labels[0,part==0] = 1
        labels[1,part==1] = 1
        nmi = calc_NMI(true_labels,labels)
        df = pd.concat([df,pd.DataFrame({'method':'grassmann','repeat':i,'centroids':[C],'nmi':nmi,'label':[labels],'noise':scale})])

    ######################################################
    # weighted grassmann clustering
    from src.DMM_EM.riemannian_clustering import weighted_grassmann_clustering
    u_in = u_all
    l_in = l_all
    for i in range(num_repeats):
        C,C_weights,part,obj = weighted_grassmann_clustering(u_in,K=2,X_weights=l_in)
        labels = np.zeros((2,num_points_per_cluster*2))
        labels[0,part==0] = 1
        labels[1,part==1] = 1
        nmi = calc_NMI(true_labels,labels)
        df = pd.concat([df,pd.DataFrame({'method':'weighted_grassmann','repeat':i,'centroids':[C],'centroids_weights':[C_weights],'nmi':nmi,'label':[labels],'noise':scale})])

    ######################################################
    # mixture models
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
        l_all = torch.tensor(l_all)
        u_all_complex = torch.tensor(u_all_complex)

    ######################################################
    # Watson (real)
    u_in = u_all[:,:,0]
    options['modelname'] = 'Watson'
    options['rank'] = 1
    for i in range(num_repeats):
        params,train_posterior,loglik_curve = train_model(data_train=u_in,L_train=None,K=2,options=options)
        train_NMI = calc_NMI(true_labels,np.double(np.array(train_posterior)))
        df = pd.concat([df,pd.DataFrame({'method':'Watson','repeat':i,'centroids':[params],'nmi':train_NMI,'label':[labels],'noise':scale})])

    ######################################################
    # Watson (complex)
    u_in = u_all_complex
    options['modelname'] = 'Complex_Watson'
    options['rank'] = 1
    for i in range(num_repeats):
        params,train_posterior,loglik_curve = train_model(data_train=u_in,L_train=None,K=2,options=options)
        train_NMI = calc_NMI(true_labels,np.double(np.array(train_posterior)))
        df = pd.concat([df,pd.DataFrame({'method':'Complex_Watson','repeat':i,'centroids':[params],'nmi':train_NMI,'label':[labels],'noise':scale})])

    ######################################################
    # ACG (real)
    u_in = u_all[:,:,0]
    options['modelname'] = 'ACG'
    options['rank'] = 'fullrank'
    for i in range(num_repeats):
        params,train_posterior,loglik_curve = train_model(data_train=u_in,L_train=None,K=2,options=options)
        train_NMI = calc_NMI(true_labels,np.double(np.array(train_posterior)))
        Lambda=params['Lambda']
        df = pd.concat([df,pd.DataFrame({'method':'ACG_fullrank','repeat':i,'centroids':[Lambda],'nmi':train_NMI,'label':[labels],'noise':scale})])

    ######################################################
    # ACG (complex)
    u_in = u_all_complex
    options['modelname'] = 'Complex_ACG'
    options['rank'] = 'fullrank'
    for i in range(num_repeats):
        params,train_posterior,loglik_curve = train_model(data_train=u_in,L_train=None,K=2,options=options)
        train_NMI = calc_NMI(true_labels,np.double(np.array(train_posterior)))
        Lambda=params['Lambda']
        df = pd.concat([df,pd.DataFrame({'method':'Complex_ACG_fullrank','repeat':i,'centroids':[Lambda],'nmi':train_NMI,'label':[labels],'noise':scale})])

    ######################################################
    # MACG
    u_in = u_all
    options['modelname'] = 'MACG'
    options['rank'] = 'fullrank'
    for i in range(num_repeats):
        params,train_posterior,loglik_curve = train_model(data_train=u_in,L_train=None,K=2,options=options)
        train_NMI = calc_NMI(true_labels,np.double(np.array(train_posterior)))
        # Sigma = params['M']@np.swapaxes(params['M'],-2,-1)+np.eye(3)
        Sigma = params['Lambda']
        df = pd.concat([df,pd.DataFrame({'method':'MACG_fullrank','repeat':i,'centroids':[Sigma],'nmi':train_NMI,'label':[labels],'noise':scale})])

    ######################################################
    # SingularWishart
    u_in = u_all
    l_in = l_all
    options['modelname'] = 'SingularWishart'
    options['rank'] = 'fullrank'
    for i in range(num_repeats):
        params,train_posterior,loglik_curve = train_model(data_train=u_in,L_train=l_in,K=2,options=options)
        train_NMI = calc_NMI(true_labels,np.double(np.array(train_posterior)))
        # Psi = params['M']@np.swapaxes(params['M'],-2,-1)+np.eye(3)
        Psi = params['Lambda']
        df = pd.concat([df,pd.DataFrame({'method':'SingularWishart_fullrank','repeat':i,'centroids':[Psi],'nmi':train_NMI,'label':[labels],'noise':scale})])

# %%
df.to_pickle('src/visualization/fits/cluster_results_noise.pkl')
