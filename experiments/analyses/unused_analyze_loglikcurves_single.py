import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.DataFrame(columns=['Log likelihood','model','Optimizer','Initialization'])

num_repl_outer = 10
K=2
p=3

inits = ['unif','++','dc']
LRs = [0,0.001,0.01,0.1]
for m in range(2):
    if m==0:
        name='Watson'
    elif m==1:
        name='ACG'
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
                file_path = 'experiments/synth_outputs/'+name+'_'+expname+'_traintestlikelihood_r'+str(rep)+'.csv'
                try:
                    like = np.loadtxt(file_path)[0]
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

plt.figure()
sns.violinplot(data=df[df.model=='Watson'],x='Initialization',y='Log likelihood',hue='Optimizer',inner='point',scale='count')
plt.title('Watson mixture, K=2')
plt.figure()
sns.violinplot(data=df[df.model=='ACG'],x='Initialization',y='Log likelihood',hue='Optimizer',inner='point',scale='count')
plt.title('Angular Central Gaussian mixture, K=2')
plt.show()
stop=7