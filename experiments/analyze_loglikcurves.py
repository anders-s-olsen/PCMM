import seaborn as sns
import numpy as np
import pandas as pd

df = pd.DataFrame(columns=['loglik','model','LR','init'])

num_repl_outer = 10
K=2

inits = ['unif','++','dc']
LRs = [0.01,0.1,1]
for m in range(4):
    if m==0:
        name='Watson_EM'
    elif m==1:
        name='ACG_EM'
    elif m==2:
        name='Watson_torch'
    elif m==3:
        name='ACG_torch'
    for init in inits:
        for LR in LRs:
            if LR==1:
                LRname = '1.0'
            else:
                LRname = str(LR)
            if m==0 or m==1: #only keep the LR=0.01 from this
                if LR>0.01:
                    continue
            expname = '3d_'+init+'_'+LRname
            
            testlike = []
            modelname = []
            LRval = []
            initval = []
            for rep in range(num_repl_outer):
                file_path = 'experiments/outputs'+expname+'/'+name+'_'+expname+'_traintestlikelihood'+str(K)+'_r'+str(rep)+'.csv'
                testlike.append(np.loadtxt(file_path)[1])
                if m==0 or m==2:
                    modelname.append('Watson')
                elif m==1 or m==3:
                    modelname.append('ACG')
                if m==0 or m==1:
                    LRval.append('EM')
                else:
                    LRval.append(str(LR))
                initval.append(init)
            dftmp = pd.DataFrame({'loglik': testlike, 'model': modelname, 'LR': LRval, 'init': initval})
            df = pd.concat([df,dftmp])

sns.violinplot(data=df[df.model=='Watson'],x='init',y='loglik',hue='LR',inner='point')
sns.violinplot(data=df[df.model=='ACG'],x='init',y='loglik',hue='LR',inner='point')

stop=7