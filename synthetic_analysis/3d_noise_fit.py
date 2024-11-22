import numpy as np
import pandas as pd
from src.helper_functions import train_model,test_model,calc_NMI
import h5py as h5

df = pd.DataFrame()
num_repeats = 10
num_points_per_cluster = 1000
K = 2
p = 3
q = 2
modelnames = ['euclidean','diametrical','complex_diametrical','grassmann','weighted_grassmann','Watson','Complex_Watson','ACG','Complex_ACG','MACG','SingularWishart','Normal']

# make noise levels from 0 to 2pi in 8 steps
levels = np.linspace(np.pi/16,np.pi,16)
true_labels = np.zeros((K,num_points_per_cluster*K))
true_labels[0,:num_points_per_cluster] = 1
true_labels[1,num_points_per_cluster:] = 1

def fill_df(df,method,repeat,centroids,train_labels,noise,true_labels,train_obj,test_labels=None,test_obj=None,HMM=False):
    if train_labels.ndim!=2: #binary, kmeans
        labels = np.zeros((2,num_points_per_cluster*2))
        labels[0,train_labels==0] = 1
        labels[1,train_labels==1] = 1
        nmi = calc_NMI(true_labels,labels)
        labels_test = np.zeros((2,num_points_per_cluster*2))
        labels_test[0,test_labels==0] = 1
        labels_test[1,test_labels==1] = 1
        nmi_test = calc_NMI(true_labels,labels_test)
    else:
        nmi = calc_NMI(true_labels,np.double(np.array(train_labels)))
        nmi_test = calc_NMI(true_labels,np.double(np.array(test_labels)))
    df = pd.concat([df,pd.DataFrame({'method':method,'repeat':repeat,'HMM':HMM,'centroids':[centroids],'train_nmi':nmi,'noise':noise,'train_obj':train_obj,'test_nmi':nmi_test,'test_obj':test_obj})],ignore_index=True)
    return df

for scale in range(len(levels)):
    print('Working on '+str(levels[scale]))
    noise_scale = levels[scale]

    # only for K=2, change if otherwise
    theta1 = np.array([0,np.pi/2,0]) #theta1 = np.array([0,np.pi/2,np.pi])
    theta2 = np.array([0,0,np.pi/2]) #theta2 = np.array([0,np.pi,np.pi/2])
    thetas = [theta1,theta2]

    u_all = np.zeros((num_points_per_cluster,p,q,K,2)) #n points, p=3, q=2, k=2, 2 for train/test
    l_all = np.zeros((num_points_per_cluster,q,K,2)) #n points, q=2, k=2, 2 for train/test
    u_all_complex = np.zeros((num_points_per_cluster,p,K,2),dtype=complex) #n points, p=3, k=2, 2 for train/test
    cos_all = np.zeros((num_points_per_cluster,p,K,2)) #n points, p=3, k=2, 2 for train/test
    for train_test in range(2):
        for n in range(num_points_per_cluster):
            for k in range(K):
                random_point = np.random.random(3)*noise_scale-noise_scale/2
                theta_random = thetas[k]+random_point
                # theta_random = np.array([theta1[0]+np.random.random()*scale,theta1[1]+np.random.random()*scale,theta1[2]+np.random.random()*scale])
                
                #timeseries
                cos_random = np.cos(theta_random)
                cos_all[n,:,k,train_test] = cos_random

                # eigenvector of cosinus matrix
                coh_map = np.outer(np.cos(theta_random),np.cos(theta_random))+np.outer(np.sin(theta_random),np.sin(theta_random))
                l,u = np.linalg.eig(coh_map)
                order = np.argsort(l)[::-1]
                u_all[n,:,0,k,train_test] = u[:,order[0]]
                l_all[n,0,k,train_test] = l[order[0]]
                u_all[n,:,1,k,train_test] = u[:,order[1]]
                l_all[n,1,k,train_test] = l[order[1]]

                coh_map_complex = np.outer(np.exp(1j*theta_random),np.exp(-1j*theta_random))
                l,u = np.linalg.eig(coh_map_complex)
                u_all_complex[n,:,k,train_test] = u[:,np.argmax(l)]

    cos_real_train = np.concatenate((cos_all[:,:,0,0],cos_all[:,:,1,0]),axis=0)
    cos_real_test = np.concatenate((cos_all[:,:,0,1],cos_all[:,:,1,1]),axis=0)
    u_real_train = np.concatenate((u_all[:,:,:,0,0],u_all[:,:,:,1,0]),axis=0)
    u_real_test = np.concatenate((u_all[:,:,:,0,1],u_all[:,:,:,1,1]),axis=0)
    l_real_train = np.concatenate((l_all[:,:,0,0],l_all[:,:,1,0]),axis=0)
    l_real_test = np.concatenate((l_all[:,:,0,1],l_all[:,:,1,1]),axis=0)
    u_complex_train = np.concatenate((u_all_complex[:,:,0,0],u_all_complex[:,:,1,0]),axis=0)
    u_complex_test = np.concatenate((u_all_complex[:,:,0,1],u_all_complex[:,:,1,1]),axis=0)

    if scale==0:
        #save the complex train data
        with h5.File('src/visualization/fits/complex_data_noise3.h5','w') as f:
            f.create_dataset('U',data=u_complex_train)

    options = {}
    # options['init'] = 'dc_seg'
    options['LR'] = 0.1
    options['tol'] = 1e-8
    options['max_iter'] = 100000
    options['num_repl_inner'] = 1
    options['threads'] = 8
    # options['HMM'] = False
    options['rank'] = 3#'fullrank'
    options['num_repl'] = 1

    for model in modelnames:
        print('Working on '+model)
        if model in ['euclidean','diametrical','Watson','ACG']:
            u_in_train = u_real_train[:,:,0]
            u_in_test = u_real_test[:,:,0]
        elif model in ['complex_diametrical','Complex_Watson','Complex_ACG']:
            u_in_train = u_complex_train
            u_in_test = u_complex_test
        elif model in ['grassmann','MACG']:
            u_in_train = u_real_train
            u_in_test = u_real_test
        elif model in ['weighted_grassmann','SingularWishart']:
            u_in_train = u_real_train*np.sqrt(l_real_train)[:,None,:]
            u_in_test = u_real_test*np.sqrt(l_real_test)[:,None,:]
        elif model in ['Normal']:
            u_in_train = cos_real_train
            u_in_test = cos_real_test
        if model in ['euclidean','diametrical','complex_diametrical','grassmann','weighted_grassmann']:
            options['init']='++'
        elif model in ['Watson','Complex_Watson','ACG','Complex_ACG']:
            options['init']='dc'
        elif model in ['MACG']:
            options['init']='gc'
        elif model in ['SingularWishart']:
            options['init']='wgc'
        elif model in ['Normal']:
            options['init']='euclidean'
            
        for HMM in [False,True]:
            if HMM and model in ['euclidean','diametrical','complex_diametrical','grassmann','weighted_grassmann']:
                continue

            options['HMM'] = HMM
            options['modelname'] = model
            for i in range(num_repeats):
                params,train_posterior,loglik_curve = train_model(data_train=u_in_train,K=K,options=options)
                test_loglik,test_posterior,_ = test_model(data_test=u_in_test,params=params,K=K,options=options)
                df = fill_df(df,model,i,params,train_posterior,scale,true_labels,train_obj=loglik_curve[-1],test_labels=test_posterior,test_obj=test_loglik,HMM=HMM)

        df.to_pickle('src/visualization/fits/cluster_results_noise4.pkl')
