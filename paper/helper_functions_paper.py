import pandas as pd
import numpy as np
import h5py as h5
from PCMM.helper_functions import train_model,test_model
import matplotlib.pyplot as plt
import seaborn as sns

def load_fMRI_data(data_file,options,remove_first_ten=False):
    assert options['modelname'] in ['Watson','ACG','MACG','SingularWishart','Complex_Watson','Complex_ACG','Normal','Complex_Normal','least_squares','diametrical','complex_diametrical','grassmann','weighted_grassmann']

    if options['dataset'] == 'all_tasks':
        with h5.File(data_file,'r') as f:
            if options['modelname'] in ['Complex_Watson','Complex_ACG','complex_diametrical']:
                # complex normalized phase vectors
                data_train = f['U_complex_train'][:]
                data_test = f['U_complex_test'][:]
            elif options['modelname'] in ['Watson','ACG','least_squares','diametrical']:
                # leading eigenvector of cosinus phase coherence matrix
                data_train = f['U_cos_train'][:][:,:,0]
                data_test = f['U_cos_test'][:][:,:,0]
            elif options['modelname'] in ['MACG','grassmann']:
                # both eigenvectors of cosinus phase coherence matrix
                data_train = f['U_cos_train'][:]
                data_test = f['U_cos_test'][:]
            elif options['modelname'] in ['SingularWishart','weighted_grassmann']:
                # both eigenvectors and both eigenvalues of cosinus phase coherence matrix
                data_train = f['U_cos_train'][:]*np.sqrt(f['L_cos_train'][:][:,None,:])
                data_test = f['U_cos_test'][:]*np.sqrt(f['L_cos_test'][:][:,None,:])
            elif options['modelname'] in ['Complex_Normal']:
                # complex normalized phase vectors scaled by hilbert amplitude
                data_train = f['U_complex_train'][:]*f['A_train'][:]
                data_test = f['U_complex_test'][:]*f['A_test'][:]
            elif options['modelname'] in ['Normal']:
                # filtered time series data (no Hilbert transform)
                data_train = f['timeseries_train'][:]
                data_test = f['timeseries_test'][:]
            else:
                raise ValueError("Problem")
        if remove_first_ten:
            num_pts_per_task = np.array([176,253,316,284,232,274,405])
            # new_num_pts_per_task = num_pts_per_task - 5
            cumsum_pts_per_task = np.concatenate([np.zeros(1),np.cumsum(num_pts_per_task)]).astype(int)
            # new_cumsum_pts_per_task = np.concatenate([np.zeros(1),np.cumsum(new_num_pts_per_task)])
            num_pts_per_subject = np.sum(num_pts_per_task)
            data_train2 = []
            data_test2 = []
            for sub in range(155):
                for task in range(7):
                    data_train2.append(data_train[sub*num_pts_per_subject+cumsum_pts_per_task[task]+10:sub*num_pts_per_subject+cumsum_pts_per_task[task+1]])
            for sub in range(99):
                for task in range(7):
                    data_test2.append(data_test[sub*num_pts_per_subject+cumsum_pts_per_task[task]+10:sub*num_pts_per_subject+cumsum_pts_per_task[task+1]])
            data_train = np.concatenate(data_train2,axis=0)
            data_test = np.concatenate(data_test2,axis=0)
        return data_train,data_test[0:2],data_test
    with h5.File(data_file,'r') as f:
        if options['modelname'] in ['Complex_Watson','Complex_ACG','complex_diametrical']:
            # complex normalized phase vectors
            data_train = f['U_complex_train'][:][:,:,0]
            data_test1 = f['U_complex_test1'][:][:,:,0]
            data_test2 = f['U_complex_test2'][:][:,:,0] 
        elif options['modelname'] in ['Watson','ACG','least_squares','diametrical']:
            # leading eigenvector of cosinus phase coherence matrix
            data_train = f['U_cos_train'][:][:,:,0]
            data_test1 = f['U_cos_test1'][:][:,:,0]
            data_test2 = f['U_cos_test2'][:][:,:,0]
        elif options['modelname'] in ['MACG','grassmann']:
            # both eigenvectors of cosinus phase coherence matrix
            data_train = f['U_cos_train'][:]
            data_test1 = f['U_cos_test1'][:]
            data_test2 = f['U_cos_test2'][:]
        elif options['modelname'] in ['SingularWishart','weighted_grassmann']:
            # both eigenvectors and both eigenvalues of cosinus phase coherence matrix
            data_train = f['U_cos_train'][:]*np.sqrt(f['L_cos_train'][:][:,None,:])
            data_test1 = f['U_cos_test1'][:]*np.sqrt(f['L_cos_test1'][:][:,None,:])
            data_test2 = f['U_cos_test2'][:]*np.sqrt(f['L_cos_test2'][:][:,None,:])
        elif options['modelname'] in ['Complex_Normal']:
            # complex normalized phase vectors scaled by hilbert amplitude
            data_train = f['U_complex_train'][:][:,:,0]*f['A_train'][:]
            data_test1 = f['U_complex_test1'][:][:,:,0]*f['A_test1'][:]
            data_test2 = f['U_complex_test2'][:][:,:,0]*f['A_test2'][:]
        elif options['modelname'] in ['Normal']:
            # filtered time series data (no Hilbert transform)
            data_train = f['timeseries_train'][:][:,:,0]
            data_test1 = f['timeseries_test1'][:][:,:,0]
            data_test2 = f['timeseries_test2'][:][:,:,0]
        else:
            raise ValueError("Problem")
    return data_train,data_test1,data_test2

# from sklearn.mixture import GaussianMixture
# gm = GaussianMixture(n_components=7, covariance_type='full', max_iter=100000, tol=1e-6, verbose=2).fit(data_train)
# train_posterior = gm.predict_proba(data_train).T
# test1_posterior = gm.predict_proba(data_test1).T
# test2_posterior = gm.predict_proba(data_test2).T

def run(data_train,data_test1,data_test2,K,df,options,params=None,suppress_output=False,inner=None,p=116):
    if options['dataset'] == 'all_tasks':
        samples_per_sequence = [[176,253,316,284,232,274,405],0,[176,253,316,284,232,274,405]]
    else:
        samples_per_sequence = [1200,1200,2400]
    params,train_posterior,loglik_curve = train_model(data_train,K=K,options=options,suppress_output=suppress_output,samples_per_sequence=samples_per_sequence[0],params=params)
    
    train_loglik,train_posterior,train_loglik_per_sample = test_model(data_test=data_train,params=params,K=K,options=options,samples_per_sequence=samples_per_sequence[0])
    test1_loglik,test1_posterior,test1_loglik_per_sample = test_model(data_test=data_test1,params=params,K=K,options=options,samples_per_sequence=samples_per_sequence[1])
    test2_loglik,test2_posterior,test2_loglik_per_sample = test_model(data_test=data_test2,params=params,K=K,options=options,samples_per_sequence=samples_per_sequence[2])
    # end_logliks = [train_loglik,test1_loglik,test2_loglik]
    posteriors = [train_posterior,test1_posterior,test2_posterior]

    if options['HMM']:
        train_ll = train_loglik_per_sample/np.sum(samples_per_sequence[0])
        test1_ll = test1_loglik_per_sample/np.sum(samples_per_sequence[1])
        test2_ll = test2_loglik_per_sample/np.sum(samples_per_sequence[2])
    else:
        train_ll,test1_ll,test2_ll = calc_ll_per_sub(train_loglik_per_sample,test1_loglik_per_sample,test2_loglik_per_sample,options)
    end_logliks = [train_ll,test1_ll,test2_ll]
    sets = ['train','test1','test2']

    if options['dataset'] == 'MOTORSOCIAL':
        pts_pr_subject_sum = np.array([279,279,279*2])
        pts_pr_subject = [[284//2,274//2],[284//2,274//2],[284,274]]
        true_labels = [np.zeros((2,pts_pr_subject_sum[0])),np.zeros((2,pts_pr_subject_sum[1])),np.zeros((2,pts_pr_subject_sum[2]))]
        for set in range(len(sets)):
            true_labels[set][0,:pts_pr_subject[set][0]] = 1
            true_labels[set][1,pts_pr_subject[set][0]:] = 1
    elif options['dataset'] == 'all_tasks':
        pts_pr_subject_sum = np.array([1940,0,1940])
        pts_pr_subject = [[0,176,253,316,284,232,274,405],[0,0,0,0,0,0,0,0],[0,176,253,316,284,232,274,405]]
        # pts_pr_subject_sum = np.array([1870,0,1870])
        # pts_pr_subject = [[0,166,243,306,274,222,264,395],[0,0,0,0,0,0,0,0],[0,166,243,306,274,222,264,395]]
        cumsum_pts_pr_subject = np.cumsum(pts_pr_subject,axis=1)
        true_labels = [np.zeros((7,pts_pr_subject_sum[0])),np.zeros((7,pts_pr_subject_sum[1])),np.zeros((7,pts_pr_subject_sum[2]))]
        for task in range(7):
            true_labels[0][task,cumsum_pts_pr_subject[0][task]:cumsum_pts_pr_subject[0][task+1]] = 1
            true_labels[1][task,cumsum_pts_pr_subject[1][task]:cumsum_pts_pr_subject[1][task+1]] = 1
            true_labels[2][task,cumsum_pts_pr_subject[2][task]:cumsum_pts_pr_subject[2][task+1]] = 1
    else:
        pts_pr_subject_sum = np.array([1200,1200,2400])

    if options['dataset'] == 'all_tasks':
        num_subs = [data_train.shape[0]//1940,0,data_test2.shape[0]//1940]
    else:
        num_subs = [data_train.shape[0]//1200,data_test1.shape[0]//1200,data_test2.shape[0]//2400]
    entries = []
    # all_test_posterior = np.zeros((K,pts_pr_subject_sum[2]))
    all_test_posterior = []
    all_train_posterior = []
    for set in range(len(sets)):
        for i in range(num_subs[set]):
            if options['dataset'] == 'MOTORSOCIAL' and K==2:
                posterior_sub = posteriors[set][:,i*pts_pr_subject_sum[set]:(i+1)*pts_pr_subject_sum[set]]
                nmi = calc_NMI(posterior_sub,true_labels[set])
                posterior_sub_binary = np.argmax(posterior_sub,axis=0)
                classification_accuracy1 = np.sum(posterior_sub_binary==true_labels[set][0])/posterior_sub_binary.shape[0]
                classification_accuracy2 = np.sum(posterior_sub_binary==true_labels[set][1])/posterior_sub_binary.shape[0]
                classification_accuracy = max(classification_accuracy1,classification_accuracy2)
            elif options['dataset'] == 'all_tasks':
                posterior_sub = np.zeros((K,pts_pr_subject_sum[set]))
                for task in range(7):
                    posterior_sub[:,cumsum_pts_pr_subject[set][task]:cumsum_pts_pr_subject[set][task+1]] = posteriors[set][:,i*pts_pr_subject_sum[set]+cumsum_pts_pr_subject[set][task]:i*pts_pr_subject_sum[set]+cumsum_pts_pr_subject[set][task+1]]
                # posterior = posteriors[set][:,i*pts_pr_subject_sum[set]:(i+1)*pts_pr_subject_sum[set]]
                nmi = calc_NMI(posterior_sub,true_labels[set])
                classification_accuracy = np.nan
            else:
                nmi = np.nan
                classification_accuracy = np.nan
                posterior_sub = posteriors[set][:,i*pts_pr_subject_sum[set]:(i+1)*pts_pr_subject_sum[set]]
            entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),
                'K':K,'p':p,'rank':options['rank'],'inner':inner,'iter':len(loglik_curve),
                'Set':sets[set],'loglik':end_logliks[set][i],'Subject':i,'NMI':nmi,'classification_accuracy':classification_accuracy,}
            entries.append(entry)
            if set==0:
                all_train_posterior.append(posterior_sub)
            if set==2:
                all_test_posterior.append(posterior_sub)

    df = pd.concat([df,pd.DataFrame(entries)],ignore_index=True)
    return params,df,np.concatenate(all_train_posterior,axis=1),np.concatenate(all_test_posterior,axis=1)
# tasks = ['EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM']

def calc_ll_per_sub(train_loglik_per_sample,test1_loglik_per_sample,test2_loglik_per_sample,options):
    if train_loglik_per_sample.shape[0]==155:
        return train_loglik_per_sample,test1_loglik_per_sample,test2_loglik_per_sample
    if options['dataset'] == 'all_tasks':
        num_subs = [155,0,99]
        num_samples_per_sub_train = train_loglik_per_sample.shape[0]//num_subs[0]
        num_samples_per_sub_test1 = test1_loglik_per_sample.shape[0]
        num_samples_per_sub_test2 = test2_loglik_per_sample.shape[0]//num_subs[2]
    else:
        num_subs = [155,155,100]
        num_samples_per_sub_train = 1200
        num_samples_per_sub_test1 = 1200
        num_samples_per_sub_test2 = 2400

    train_loglik_per_sub = np.zeros(num_subs[0])
    test1_loglik_per_sub = np.zeros(num_subs[1])
    test2_loglik_per_sub = np.zeros(num_subs[2])

    for i in range(np.max(num_subs)):
        train_loglik_per_sub[i] = np.mean(train_loglik_per_sample[i*num_samples_per_sub_train:(i+1)*num_samples_per_sub_train])
        if i<num_subs[1]:
            test1_loglik_per_sub[i] = np.mean(test1_loglik_per_sample[i*num_samples_per_sub_test1:(i+1)*num_samples_per_sub_test1])
        if i<num_subs[2]:
            test2_loglik_per_sub[i] = np.mean(test2_loglik_per_sample[i*num_samples_per_sub_test2:(i+1)*num_samples_per_sub_test2])

    return train_loglik_per_sub,test1_loglik_per_sub,test2_loglik_per_sub
        
def calc_MI(Z1,Z2):
    P=Z1@Z2.T
    PXY=P/np.sum(P)
    PXPY=np.outer(np.sum(PXY,axis=1),np.sum(PXY,axis=0))
    ind=np.where(PXY>0) #PXY should always be >0
    MI=np.sum(PXY[ind]*np.log(PXY[ind]/PXPY[ind]))
    return MI

def calc_NMI(Z1,Z2):
    #Z1 and Z2 are two partition matrices of size (K1xN) and (K2xN) where K is number of components and N is number of samples
    NMI = (2*calc_MI(Z1,Z2))/(calc_MI(Z1,Z1)+calc_MI(Z2,Z2))
    return NMI

def make_true_mat(num_subs=10,K=5):
    rows = []
    for _ in range(num_subs):
        row = np.zeros((K,1200),dtype=bool)
        num_samples_per_cluster = 1200//K
        for k in range(K):
            row[k,num_samples_per_cluster*k:num_samples_per_cluster*(k+1)] = True
        rows.append(row)
    return np.hstack(rows)

def horizontal_boxplot(df_fig,type=1,ranks=[1,10,25]):
    order = ['Mixture: Complex Watson',
        *['Mixture: Complex ACG rank='+str(rank) for rank in ranks],
        'K-means: Complex diametrical','space1','space2',
        *['Mixture: MACG rank='+str(rank) for rank in ranks],
        *['Mixture: Singular Wishart rank='+str(rank) for rank in ranks],
        'K-means: Grassmann','K-means: Weighted Grassmann','space3','space4',
        'Mixture: Watson',
        *['Mixture: ACG rank='+str(rank) for rank in ranks],
        'K-means: Diametrical','K-means: Least squares (sign-flip)','space5','space6',
        *['Mixture: Gaussian rank='+str(rank) for rank in ranks],'space7','space8',
        *['Mixture: Complex Gaussian rank='+str(rank) for rank in ranks]]
    palette_husl = sns.color_palette("husl", n_colors=11, desat=1)
    palette_husl.append((0.5,0.5,0.5))
    palette_husl.append((0.3,0.3,0.3))
    palette_husl2 = [palette_husl[0]]+[palette_husl[1]]*len(ranks)+[palette_husl[2]]+[palette_husl[-1]]*2+[palette_husl[3]]*len(ranks)+[palette_husl[4]]*len(ranks)+[palette_husl[5]]+[palette_husl[6]]+[palette_husl[-1]]*2+[palette_husl[7]]+[palette_husl[8]]*len(ranks)+[palette_husl[9]]+[palette_husl[10]]+[palette_husl[-1]]*2+[palette_husl[11]]*len(ranks)+[palette_husl[-1]]*2+[palette_husl[12]]*len(ranks)
    # df_fig = df[df['Set']=='Out-of-sample test']
    for i in range(1,9):
        df_fig2 = pd.concat([df_fig,pd.DataFrame({'NMI':[np.nan],'names2':['space'+str(i)]}, index=[0])], ignore_index=True)
    fig = plt.figure(figsize=(10,7))
    if type == 1:
        sns.boxplot(x='NMI', y='names2', data=df_fig2, palette=palette_husl2, order=order)
        plt.xlabel('Normalized mutual information')
        plt.xlim([-0.01,1.01])
        xtitlepos = -0.02
    else:
        sns.boxplot(x='classification_accuracy', y='names2', data=df_fig2, palette=palette_husl2, order=order)
        plt.xlabel('Classification accuracy')
        plt.xlim([0.49,1.01])
        xtitlepos = 0.48
    plt.ylabel('')

    # add extra text next to y-ticks that aren't there
    ticks_per_group = np.array([2+len(ranks), 2+2*len(ranks), 3+len(ranks), len(ranks), len(ranks)])
    additional_ticks = np.concatenate([[0],np.cumsum(ticks_per_group+2)])
    # np.concatenate([np.arange(2+len(ranks)),np.arange(2+2*len(ranks))+7,np.arange(3+len(ranks))+17,np.arange(len(ranks))+25,np.arange(len(ranks))+30])
    ticks_list = [np.arange(ticks_per_group[i]) + additional_ticks[i] for i in range(len(ticks_per_group))]
    ticks_list = np.concatenate(ticks_list)
    # print(ticks_list)
    # print(np.concatenate([np.arange(2+len(ranks)),np.arange(2+2*len(ranks))+7,np.arange(3+len(ranks))+17,np.arange(len(ranks))+25,np.arange(len(ranks))+30]))
    # additional_ticks = [0, 7, 17, 25, 30]
    # plt.yticks(np.concatenate([np.arange(2+len(ranks)),np.arange(2+2*len(ranks))+7,np.arange(3+len(ranks))+17,np.arange(len(ranks))+25,np.arange(len(ranks))+30]),fontsize=8)

    if len(ranks)==3:
        ytitlepos = [-0.7,6.3,16.3,24.3,29.3]
        plt.yticks(ticks_list, fontsize=8)
    elif len(ranks)==5:
        ytitlepos = [-0.7,8.3,22.3,32.3,39.3]
        plt.yticks(ticks_list, fontsize=7)
        plt.ylim([46,-2])
    elif len(ranks)==6:
        ytitlepos = [-0.7,9.3,25.3,36.3,44.3]
        plt.yticks(ticks_list, fontsize=6)
        plt.ylim([51,-2])
        
    plt.text(xtitlepos, ytitlepos[0], 'Complex-valued phase coupling', fontsize=8,fontweight='bold', ha='right')
    plt.text(xtitlepos, ytitlepos[1], 'Cosine phase coupling', fontsize=8,fontweight='bold', ha='right')
    plt.text(xtitlepos, ytitlepos[2], 'LEiDA (reduced cosine phase coupling)', fontsize=8,fontweight='bold', ha='right')
    plt.text(xtitlepos, ytitlepos[3], 'Amplitude coupling', fontsize=8,fontweight='bold', ha='right')
    plt.text(xtitlepos, ytitlepos[4], 'Phase-amplitude coupling', fontsize=8,fontweight='bold', ha='right')

    # change the line styles
    styles = ['-']+['-']*len(ranks)+['--']+[':']+['-']*len(ranks)+['-']*len(ranks)+['--']*2+['-']+['-']*len(ranks)+['--']*2+['-']*len(ranks)+['-']*len(ranks)
    #repeat every element of styles six times
    styles2 = [item for item in styles for i in range(5)]
    l = 0
    for i,artist in enumerate(plt.gca().get_children()):
        if isinstance(artist, plt.Line2D):
            #if linestyle is not none
            if artist.get_linestyle() != 'None':
                artist.set_linestyle(styles2[l])
                l+=1
        # print(l)
    # plt.savefig(savename, bbox_inches='tight', dpi=300)
    return fig

def run_phaserando(data_train,data_test1,data_test2,K,P,df,options,params=None,suppress_output=False,inner=None,p=116):
    params,train_posterior,loglik_curve = train_model(data_train,K=K,options=options,suppress_output=suppress_output,samples_per_sequence=1200,params=params)
    test1_loglik,test1_posterior,_ = test_model(data_test=data_test1,params=params,K=K,options=options,samples_per_sequence=1200)
    test2_loglik,test2_posterior,_ = test_model(data_test=data_test2,params=params,K=K,options=options,samples_per_sequence=1200)
    train_NMI = calc_NMI(P,np.double(np.array(train_posterior)))
    test1_NMI = calc_NMI(P,np.double(np.array(test1_posterior)))
    test2_NMI = calc_NMI(P,np.double(np.array(test2_posterior)))
    entry = {'modelname':options['modelname'],'init_method':options['init'],'LR':options['LR'],'HMM':str(options['HMM']),'K':K,'p':p,'rank':options['rank'],'inner':inner,'iter':len(loglik_curve),
            'train_loglik':loglik_curve[-1],'test1_loglik':test1_loglik,'test2_loglik':test2_loglik,
            'train_NMI':train_NMI,'test1_NMI':test1_NMI,'test2_NMI':test2_NMI}
    df = pd.concat([df,pd.DataFrame([entry])],ignore_index=True)
    df.to_csv(options['outfolder']+'/'+options['experiment_name']+'.csv')
    return params,df