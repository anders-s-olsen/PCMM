import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from PCMM.mixture_EM_loop import mixture_EM_loop
from PCMM.PCMMnumpy import Watson as Watson_numpy
from PCMM.PCMMnumpy import ACG as ACG_numpy
from PCMM.PCMMnumpy import MACG as MACG_numpy
from PCMM.PCMMnumpy import SingularWishart as SingularWishart_numpy
from PCMM.PCMMnumpy import Normal as Normal_numpy
from PCMM.phase_coherence_kmeans import *

from PCMM.mixture_torch_loop import mixture_torch_loop
from PCMM.PCMMtorch import Watson as Watson_torch
from PCMM.PCMMtorch import ACG as ACG_torch
from PCMM.PCMMtorch import MACG as MACG_torch
from PCMM.PCMMtorch import SingularWishart as SingularWishart_torch
from PCMM.PCMMtorch import Normal as Normal_torch

def train_model(data_train,K,options,params=None,suppress_output=False,samples_per_sequence=0):
    p = data_train.shape[1]
    if options['modelname'] in ['Watson','Complex_Watson','ACG','Complex_ACG','MACG','SingularWishart','Normal','Complex_Normal']:
        if options['rank']=='fullrank':
            rank=0
        elif options['rank']=='lowrank': #assume full rank in lowrank setting
            rank=p
        else: 
            rank=options['rank']
        if options['LR']!=0:
            # data_train = torch.tensor(data_train)
            data_train = torch.from_numpy(data_train)

    if options['modelname'] == 'Watson':
        if options['LR']==0:
            model = Watson_numpy(K=K,p=p,params=params)
        else:
            model = Watson_torch(K=K,p=p,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'Complex_Watson':
        if options['LR']==0:
            model = Watson_numpy(K=K,p=p,complex=True,params=params)
        else:
            model = Watson_torch(K=K,p=p,complex=True,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'ACG':
        if options['LR']==0:
            model = ACG_numpy(K=K,p=p,rank=rank,params=params)
        else:
            model = ACG_torch(K=K,p=p,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'Complex_ACG':
        if options['LR']==0:
            model = ACG_numpy(K=K,p=p,rank=rank,complex=True,params=params)
        else:
            model = ACG_torch(K=K,p=p,rank=rank,complex=True,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'MACG':
        if options['LR']==0:
            model = MACG_numpy(K=K,p=p,q=2,rank=rank,params=params)
        else:
            model = MACG_torch(K=K,p=p,q=2,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence) 
    elif options['modelname'] == 'SingularWishart':
        if options['LR']==0:
            model = SingularWishart_numpy(K=K,p=p,q=2,rank=rank,params=params)
        else:
            model = SingularWishart_torch(K=K,p=p,q=2,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence) 
    elif options['modelname'] == 'Normal':
        if options['LR']==0:
            model = Normal_numpy(K=K,p=p,rank=rank,params=params)
        else:
            model = Normal_torch(K=K,p=p,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'Complex_Normal':
        if options['LR']==0:
            model = Normal_numpy(K=K,p=p,rank=rank,params=params,complex=True)
        else:
            model = Normal_torch(K=K,p=p,rank=rank,params=params,complex=True,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
    elif options['modelname'] == 'least_squares':
        C,labels,obj = least_squares_sign_flip(data_train,K=K,max_iter=options['max_iter'],num_repl=options['num_repl'],init=options['init'],tol=options['tol'])
        # X = data_train
        # X[(X>0).sum(axis=1)>p/2] = -X[(X>0).sum(axis=1)>p/2]
        # C,labels = kmeans2(X,k=K,minit=options['init'])
        labels = np.eye(K)[labels].T
        params = {'C':C}
        #euclidean distance
        # sim = -np.sum((X[:,None]-params['C'][None])**2,axis=-1)
        # obj = np.mean(np.max(sim,axis=1))
        return params,labels,obj
    elif options['modelname'] in ['diametrical','complex_diametrical']:
        C,labels,obj = diametrical_clustering(data_train,K=K,max_iter=options['max_iter'],num_repl=options['num_repl'],init=options['init'],tol=options['tol'])
        labels = np.eye(K)[labels].T
        params = {'C':C}
        return params,labels,obj
    elif options['modelname'] == 'grassmann':
        C,labels,obj = grassmann_clustering(data_train,K=K,max_iter=options['max_iter'],num_repl=options['num_repl'],init=options['init'],tol=options['tol'])
        labels = np.eye(K)[labels].T
        params = {'C':C}
        return params,labels,obj
    elif options['modelname'] == 'weighted_grassmann':
        C,labels,obj = weighted_grassmann_clustering(data_train,K=K,max_iter=options['max_iter'],num_repl=options['num_repl'],init=options['init'],tol=options['tol'])
        labels = np.eye(K)[labels].T
        params = {'C':C}
        return params,labels,obj
    else:
        raise ValueError("Problem")
        
    #if the element 'tol' doesn't exist in options, set it to default 1e-10
    if 'tol' not in options:
        options['tol'] = 1e-10
    if 'max_iter' not in options:
        options['max_iter'] = 100000
    if 'num_repl' not in options:
        options['num_repl'] = 1
    if 'init' not in options:
        raise ValueError('Please provide an initialization method')
    if 'LR' not in options:
        options['LR'] = 0
    if 'threads' not in options:
        options['threads'] = 8
    if 'decrease_lr_on_plateau' not in options:
        options['decrease_lr_on_plateau'] = False
    if 'num_comparison' not in options:
        options['num_comparison'] = 50

    if options['LR']==0: #EM
        params,posterior,loglik = mixture_EM_loop(model,data_train,tol=options['tol'],max_iter=options['max_iter'],
                                                num_repl=options['num_repl'],init=options['init'],
                                                suppress_output=suppress_output)
    else:
        params,posterior,loglik = mixture_torch_loop(model,data_train,tol=options['tol'],max_iter=options['max_iter'],
                                        num_repl=options['num_repl'],init=options['init'],LR=options['LR'],
                                        suppress_output=suppress_output,threads=options['threads'],decrease_lr_on_plateau=options['decrease_lr_on_plateau'],num_comparison=options['num_comparison'])
    return params,posterior,loglik
    
def test_model(data_test,params,K,options,samples_per_sequence=0):
    p = data_test.shape[1]
    # if rank is a key in options
    if 'rank' in options:
        if options['rank']=='fullrank':
            rank=0
        elif options['rank']=='lowrank': #assume full rank in lowrank setting
            rank=p
        else: 
            rank=options['rank']
    
    if options['modelname'] in ['Watson','Complex_Watson','ACG','Complex_ACG','MACG','SingularWishart','Normal','Complex_Normal']:
        if options['LR']!=0:
            # data_test = torch.tensor(data_test)
            data_test = torch.from_numpy(data_test)
            
        if options['modelname'] == 'Watson':    
            if options['LR']==0:
                model = Watson_numpy(K=K,p=p,params=params)
            else:
                model = Watson_torch(K=K,p=p,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'Complex_Watson':
            if options['LR']==0:
                model = Watson_numpy(K=K,p=p,complex=True,params=params)
            else:
                model = Watson_torch(K=K,p=p,complex=True,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'ACG':
            if options['LR']==0:
                model = ACG_numpy(K=K,p=p,rank=rank,params=params)
            else:
                model = ACG_torch(K=K,p=p,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'Complex_ACG':
            if options['LR']==0:
                model = ACG_numpy(K=K,p=p,complex=True,rank=rank,params=params)
            else:
                model = ACG_torch(K=K,p=p,complex=True,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'MACG':
            if options['LR']==0:
                model = MACG_numpy(K=K,p=p,q=2,rank=rank,params=params)
            else:
                model = MACG_torch(K=K,p=p,q=2,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'SingularWishart':
            if options['LR']==0:
                model = SingularWishart_numpy(K=K,p=p,q=2,rank=rank,params=params)
            else:
                model = SingularWishart_torch(K=K,p=p,q=2,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'Normal':
            if options['LR']==0:
                model = Normal_numpy(K=K,p=p,rank=rank,params=params)
            else:
                model = Normal_torch(K=K,p=p,rank=rank,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
        elif options['modelname'] == 'Complex_Normal':
            if options['LR']==0:
                model = Normal_numpy(K=K,p=p,rank=rank,complex=True,params=params)
            else:
                model = Normal_torch(K=K,p=p,rank=rank,complex=True,params=params,HMM=options['HMM'],samples_per_sequence=samples_per_sequence)
        test_loglik, test_loglik_per_sample = model.test_log_likelihood(X=data_test)
        test_posterior = model.posterior(X=data_test)
    elif options['modelname'] in ['least_squares','diametrical','complex_diametrical','grassmann','weighted_grassmann']:
        if options['modelname'] == 'least_squares':
            #eucdliean distance
            X=data_test
            X[(X>0).sum(axis=1)>p/2] = -X[(X>0).sum(axis=1)>p/2]
            sim = -np.sum((X[:,None]-params['C'][None])**2,axis=-1)
        elif options['modelname'] in ['diametrical','complex_diametrical']:
            sim = np.abs(data_test@params['C'].conj().T)**2
        elif options['modelname'] == 'grassmann':
            sim = -1/np.sqrt(2)*(2*data_test.shape[2]-2*np.linalg.norm(np.swapaxes(data_test[:,None],-2,-1)@params['C'][None],axis=(-2,-1))**2)
        elif options['modelname'] == 'weighted_grassmann':
            C_weights = np.linalg.norm(params['C'],axis=1)**2
            L_test = np.linalg.norm(data_test,axis=1)**2
            B = np.swapaxes(data_test,-2,-1)[:,None]@(params['C'][None])
            sim = -1/np.sqrt(2)*(np.sum(L_test**2,axis=-1)[:,None]+np.sum(C_weights**2,axis=-1)[None]-2*np.linalg.norm(B,axis=(-2,-1))**2)#
        
        test_loglik = np.mean(np.max(sim,axis=1))
        test_loglik_per_sample = np.max(sim,axis=1)
        
        test_posterior = np.argmax(sim,axis=1) # assign each point to the cluster with the highest similarity
        test_posterior = np.eye(K)[test_posterior].T
    else:
        raise ValueError("Problem, modelname:",options['modelname'])
    
    return test_loglik.item(),test_posterior,test_loglik_per_sample
        
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
    plt.text(xtitlepos, ytitlepos[2], 'LEiDA (leading eigenvector of cos. phase coupling)', fontsize=8,fontweight='bold', ha='right')
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