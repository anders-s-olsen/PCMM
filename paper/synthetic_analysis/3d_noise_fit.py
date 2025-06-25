import numpy as np
import pandas as pd
from PCMM.helper_functions import train_model,test_model,calc_NMI
import h5py as h5

def fill_df(df,method,repeat,train_labels,noise,true_labels,train_obj,test_labels=None,test_obj=None,HMM=False):
    num_points_per_cluster = 1000
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
    df = pd.concat([df,pd.DataFrame({'method':method,'repeat':repeat,'HMM':HMM,'train_nmi':[nmi],'noise':[noise],'train_obj':[train_obj],'test_nmi':[nmi_test],'test_obj':[test_obj]})],ignore_index=True)
    return df

def run_experiment(modelname):
    try:
        df = pd.read_csv('paper/synthetic_analysis/fits/cluster_results_noise_'+modelname+'.csv')
        num_done = df['level'].max()
        df = df[df['level']!=num_done]
    except:
        df = pd.DataFrame()
        num_done = 0
    num_repeats = 5
    num_points_per_cluster = 1000
    K = 2
    p = 3
    q = 2

    # make noise levels from 0 to 2pi in 8 steps
    levels = np.linspace(np.pi/16,3*np.pi/2,24)
    true_labels = np.zeros((K,num_points_per_cluster*K))
    true_labels[0,:num_points_per_cluster] = 1
    true_labels[1,num_points_per_cluster:] = 1
    for scale in range(num_done,len(levels)):

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
        analytic_signal_all = np.zeros((num_points_per_cluster,p,K,2),dtype=complex) #n points, p=3, k=2, 2 for train/test
        for train_test in range(2):
            for n in range(num_points_per_cluster):
                for k in range(K):
                    random_point = np.random.random(3)*noise_scale-noise_scale/2
                    theta_random = thetas[k]+random_point
                    # theta_random = np.array([theta1[0]+np.random.random()*scale,theta1[1]+np.random.random()*scale,theta1[2]+np.random.random()*scale])
                    
                    #timeseries
                    cos_all[n,:,k,train_test] = np.cos(theta_random)

                    # eigenvector of cosinus matrix
                    coh_map = np.outer(np.cos(theta_random),np.cos(theta_random))+np.outer(np.sin(theta_random),np.sin(theta_random))
                    l,u = np.linalg.eig(coh_map)
                    order = np.argsort(l)[::-1]
                    u_all[n,:,0,k,train_test] = u[:,order[0]]
                    l_all[n,0,k,train_test] = l[order[0]]
                    u_all[n,:,1,k,train_test] = u[:,order[1]]
                    l_all[n,1,k,train_test] = l[order[1]]

                    # coh_map_complex = np.outer(np.exp(1j*theta_random),np.exp(-1j*theta_random))
                    # l,u = np.linalg.eig(coh_map_complex)
                    # u_all_complex[n,:,k,train_test] = u[:,np.argmax(l)]
                    u_all_complex[n,:,k,train_test] = 1/np.sqrt(3)*np.exp(1j*theta_random)

                    analytic_signal_all[n,:,k,train_test] = np.exp(1j*theta_random)

        cos_real_train = np.concatenate((cos_all[:,:,0,0],cos_all[:,:,1,0]),axis=0)
        cos_real_test = np.concatenate((cos_all[:,:,0,1],cos_all[:,:,1,1]),axis=0)
        u_real_train = np.concatenate((u_all[:,:,:,0,0],u_all[:,:,:,1,0]),axis=0)
        u_real_test = np.concatenate((u_all[:,:,:,0,1],u_all[:,:,:,1,1]),axis=0)
        l_real_train = np.concatenate((l_all[:,:,0,0],l_all[:,:,1,0]),axis=0)
        l_real_test = np.concatenate((l_all[:,:,0,1],l_all[:,:,1,1]),axis=0)
        u_complex_train = np.concatenate((u_all_complex[:,:,0,0],u_all_complex[:,:,1,0]),axis=0)
        u_complex_test = np.concatenate((u_all_complex[:,:,0,1],u_all_complex[:,:,1,1]),axis=0)
        analytic_signal_train = np.concatenate((analytic_signal_all[:,:,0,0],analytic_signal_all[:,:,1,0]),axis=0)
        analytic_signal_test = np.concatenate((analytic_signal_all[:,:,0,1],analytic_signal_all[:,:,1,1]),axis=0)

        options = {}
        # options['init'] = 'dc_seg'
        options['LR'] = 0.1
        options['tol'] = 1e-10
        options['max_iter'] = 100000
        options['num_repl_inner'] = 1
        # options['HMM'] = False
        if modelname in ['least_squares','diametrical','complex_diametrical','grassmann','weighted_grassmann','Watson','Complex_Watson','Complex_ACG']:
            options['rank'] = 1
        elif modelname in ['Normal']:
            options['rank'] = 'fullrank'
            options['LR'] = 0
        else:
            options['rank'] = 2#'fullrank'
        options['num_repl'] = 1

        # for model in modelnames:
        print('Working on '+modelname)
        if modelname in ['least_squares','diametrical','Watson','ACG']:
            u_in_train = u_real_train[:,:,0]
            u_in_test = u_real_test[:,:,0]
        elif modelname in ['complex_diametrical','Complex_Watson','Complex_ACG']:
            u_in_train = u_complex_train
            u_in_test = u_complex_test
        elif modelname in ['grassmann','MACG']:
            u_in_train = u_real_train
            u_in_test = u_real_test
        elif modelname in ['weighted_grassmann','SingularWishart']:
            u_in_train = u_real_train*np.sqrt(l_real_train)[:,None,:]
            u_in_test = u_real_test*np.sqrt(l_real_test)[:,None,:]
        elif modelname in ['Normal']:
            u_in_train = cos_real_train
            u_in_test = cos_real_test
        elif modelname in ['Complex_Normal']:
            u_in_train = analytic_signal_train
            u_in_test = analytic_signal_test
    
        for i in range(num_repeats):
            if modelname in ['least_squares','diametrical','complex_diametrical','grassmann','weighted_grassmann']:
                options['init']='++'
            elif modelname in ['Watson','Complex_Watson','ACG','Complex_ACG']:
                options['init']='dc'
            elif modelname in ['MACG']:
                options['init']='gc'
            elif modelname in ['SingularWishart']:
                options['init']='wgc'
            elif modelname in ['Normal','Complex_Normal']:
                options['init']='ls'
            HMM = False
            # for HMM in [False,True]:
            if HMM and modelname in ['least_squares','diametrical','complex_diametrical','grassmann','weighted_grassmann']:
                continue

            options['HMM'] = HMM
            options['modelname'] = modelname
            if HMM:
                options['init'] = 'no'
                params,train_posterior,loglik_curve = train_model(data_train=u_in_train,K=K,options=options,params=params,suppress_output=True)    
            else:
                params,train_posterior,loglik_curve = train_model(data_train=u_in_train,K=K,options=options,params=None,suppress_output=True)    
            test_loglik,test_posterior,_ = test_model(data_test=u_in_test,params=params,K=K,options=options)
            df = fill_df(df,modelname,i,train_posterior,scale,true_labels,train_obj=loglik_curve[-1],test_labels=test_posterior,test_obj=test_loglik,HMM=HMM)

            df.to_csv('paper/synthetic_analysis/fits/cluster_results_noise_'+modelname+'.csv')

if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        print(sys.argv)
        run_experiment(modelname=sys.argv[1])
    else:
        modelnames = ['least_squares','diametrical','complex_diametrical','grassmann','weighted_grassmann','Watson','Complex_Watson','ACG','Complex_ACG','MACG','SingularWishart','Normal', 'Complex_Normal']
        modelnames = ['Normal']
        for modelname in modelnames:
            run_experiment(modelname=modelname)