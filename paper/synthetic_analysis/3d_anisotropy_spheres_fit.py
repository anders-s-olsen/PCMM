# %%
#%%
import numpy as np
import h5py as h5
import pandas as pd
df = pd.DataFrame()
num_repeats = 5
scale = 0.2
# scale2 = 3*np.pi/4
num_points_per_cluster = 1000

# %%
# theta1 = np.array([0,0,np.pi])
# theta2 = np.array([0,np.pi,0])

possible_phases = np.linspace(0,2*np.pi,1000)

u_all1 = np.zeros((num_points_per_cluster,3,2))
u_all2 = np.zeros((num_points_per_cluster,3,2))

n = num_points_per_cluster
t = np.linspace(0, 5, n)
for traintest in range(2):
    # for n in range(num_points_per_cluster):
    noises = np.zeros((3,n))
    noises[2] = np.random.uniform(-scale/2, scale/2, n)
    noises[2] = np.random.standard_normal(n)*scale/4
    signal6 = np.zeros((3,n))
    signal6[0] = np.cos(2*np.pi*t)
    signal6[1] = np.sin(2*np.pi*t)

    cos_signal6 = signal6+noises
    u_all1[:,:,traintest] = (cos_signal6/np.linalg.norm(cos_signal6,axis=0)).T
    
    noises = np.zeros((3,n))
    noises[1] = np.random.uniform(-scale/2, scale/2, n)
    noises[1] = np.random.standard_normal(n)*scale/4
    signal7 = np.zeros((3,n))
    signal7[0] = np.cos(2*np.pi*t)
    signal7[2] = np.sin(2*np.pi*t)

    cos_signal7 = signal7+noises
    u_all2[:,:,traintest] = (cos_signal7/np.linalg.norm(cos_signal7,axis=0)).T
        
u_all_train = np.concatenate((u_all1[:,:,0],u_all2[:,:,0]),axis=0)
u_all_test = np.concatenate((u_all1[:,:,1],u_all2[:,:,1]),axis=0)
true_labels = np.zeros((2,num_points_per_cluster*2))
true_labels[0,:num_points_per_cluster] = 1
true_labels[1,num_points_per_cluster:] = 1

# save also the data
np.save('paper/synthetic_analysis/fits/cluster_data.npy',u_all_train)
with h5.File('paper/synthetic_analysis/fits/cluster_data.h5', 'w') as f:
    f.create_dataset('data', data=u_all_train)
    # f.create_dataset('evs', data=l_all)


from PCMM.helper_functions import train_model,test_model
options = {}
options['init'] = 'dc'
options['LR'] = 0.1
options['tol'] = 1e-10
options['max_iter'] = 10000
options['num_repl_inner'] = 1
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
    # options['modelname'] = 'Watson'
    # options['threads'] = 8
    # options['rank'] = 1
    # for K in range(1,11):
    #     params,train_posterior,loglik_curve = train_model(data_train=u_in,K=K,options=options)
    #     test_loglik,test_posterior,_ = test_model(data_test=u_in_test,params=params,K=K,options=options)
    #     np.savetxt('paper/synthetic_analysis/fits/watson_centroids_mu_K='+str(K)+'.txt',params['mu'])
    #     np.savetxt('paper/synthetic_analysis/fits/watson_centroids_kappa_K='+str(K)+'.txt',params['kappa'])
    #     np.savetxt('paper/synthetic_analysis/fits/watson_centroids_pi_K='+str(K)+'.txt',params['pi'])
    #     df = pd.concat([df,pd.DataFrame([{'modelname':options['modelname'],'K':K,'train_loglik':loglik_curve[-1],'test_loglik':test_loglik,'inner':inner}])],ignore_index=True)


    for rank in [1,2,3]:
        options['modelname'] = 'ACG'
        # options['rank'] = 'fullrank'
        options['rank'] = rank
        for K in range(1,11):
            if rank==3 and K==6:
                stophere=8
            print('rank:',rank,'K:',K)
            params,train_posterior,loglik_curve = train_model(data_train=u_in,K=K,options=options)
            test_loglik,test_posterior,_ = test_model(data_test=u_in_test,params=params,K=K,options=options)
            for k in range(K):
                Lambda_k = params['M'][k]@params['M'][k].T+np.eye(3)
                Lambda_k = Lambda_k/np.trace(Lambda_k)
                np.savetxt('paper/synthetic_analysis/fits/centroids/ACG_centroids_K='+str(K)+str(k)+'_rank='+str(rank)+'.txt',Lambda_k)
                # np.savetxt('paper/synthetic_analysis/fits/ACG_centroids_K='+str(K)+str(k)+'.txt',params['Lambda'][k])
            pi = torch.nn.functional.softmax(params['pi'],dim=0).detach().numpy()
            np.savetxt('paper/synthetic_analysis/fits/centroids/ACG_centroids_pi_K='+str(K)+'_rank='+str(rank)+'.txt',pi)
            # np.savetxt('paper/synthetic_analysis/fits/ACG_labels_K='+str(K)+'.txt',df[df['method']=='ACG_fullrank'].label.iloc[0])
            df = pd.concat([df,pd.DataFrame([{'modelname':options['modelname'],'K':K,'train_loglik':loglik_curve[-1],'test_loglik':test_loglik,'inner':inner,'rank':rank}])],ignore_index=True)
            if K==2:
                stophere=8
    df.to_pickle('paper/synthetic_analysis/fits/cluster_results_rank.pkl')