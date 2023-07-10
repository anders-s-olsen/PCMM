import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sns.set()

num_repl_outer = 10
inits = ['unif','++','dc']
LRs = [0,0.001,0.01,0.1]
for m in range(2):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    if m==0:
        name='Watson'
    elif m==1:
        name='ACG'
    left_labels = ['Log-likelihood curves, p=3','Log-likelihood curves, p=10','Log-likelihood curves, p=25']
    top_labels = ['Number of components, K=2','Number of components, K=5','Number of components, K=10']
    for idx1,p in enumerate([3,10,25]):
        for idx2,K in enumerate([2,5,10]):
            if K>p:
                continue
            df = pd.DataFrame(columns=['Log likelihood','model','Optimizer','Initialization'])
            for init in inits:
                for LR in LRs:
                    if LR==1 or LR==0:
                        LRname = str(LR)+'.0'
                    else:
                        LRname = str(LR)
                    expname = '3d_'+init+'_'+LRname+'_p'+str(p)+'_K'+str(K)
                    
                    testlike = []
                    modelname = []
                    LRval = []
                    initval = []
                    for rep in range(num_repl_outer):
                        file_path = 'experiments/outputs'+expname+'/'+name+'_'+expname+'_traintestlikelihood'+str(K)+'_r'+str(rep)+'.csv'
                        try:
                            like = np.loadtxt(file_path)[0]
                            # if like < -1800:
                            #     raise ValueError
                            testlike.append(like)
                        except: continue
                        modelname.append(name)
                        if LR==0:
                            LRval.append('EM')
                        else:
                            LRval.append('PyTorch, LR='+str(LR))
                        if init == 'unif':
                            initval.append('Random uniform')
                        elif init == '++':
                            initval.append('Diametrical clustering ++')
                        elif init == 'dc':
                            initval.append('Diametrical clustering')
                    dftmp = pd.DataFrame({'Log likelihood': testlike, 'model': modelname, 'Optimizer': LRval, 'Initialization': initval})
                    df = pd.concat([df,dftmp])
            sns.violinplot(ax=axs[idx1,idx2],data=df[df.model==name],x='Initialization',y='Log likelihood',hue='Optimizer',inner='point',scale='count')
fig.suptitle(name+' mixture model')
plt.show()
stop=7