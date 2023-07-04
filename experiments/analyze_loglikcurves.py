import seaborn as sns
import numpy as np

num_repl_outer = 10
K=2

inits = ['unif','++','dc']
LRs = [0.01,0.1,1]
for init in inits:
    for LR in LRs:
        for m in range(4):
            expname = '3d_'+init+'_'+str(LR)
            if m==0:
                name='Watson_EM'
            elif m==1:
                name='ACG_EM'
            elif m==2:
                name='Watson_torch'
            elif m==3:
                name='ACG_torch'
               
            testlike = np.zeros(num_repl_outer)
            for rep in range(num_repl_outer):
                testlike[rep] = np.loadtxt('experiments/outputs'+expname+'/'+name+'_'+expname+'_traintestlikelihood'+str(K)+'_r'+str(rep)+'.csv')[1]

stop=7