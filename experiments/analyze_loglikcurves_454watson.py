import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
# matplotlib.rcParams.update({'font.size': 10})

num_repl_outer = 10
inits = ['unif','++','dc']
LRs = ['0.01','0.1','0.0','1.0']
exptype = '454'
# left_labels = ['Number of components, K=2','Number of components, K=5','Number of components, K=10']
fix,axs = plt.subplots(1,2,figsize=(16,9))
df = pd.DataFrame(columns=['Log likelihood','Optimizer','Initialization','Test or train','Model order (K)'])
for idx1,K in enumerate(np.arange(2,31)):
    for init in inits:
        for LRidx,LR in enumerate(LRs):
            expname = '454_full_'+init+'_'+LR+'_p454_K'+str(K)
            testlike = []
            LRval = []
            initval = []
            testtrainval = []
            modelorder = []
            for rep in range(num_repl_outer):
                for tt,traintest in enumerate(['Train','Test']):
                    file_path = 'experiments/'+exptype+'_outputs/Watson_'+expname+'_traintestlikelihood_r'+str(rep)+'.csv'
                    try:
                        like = np.loadtxt(file_path)[tt]
                        testlike.append(like)
                    except: continue
                    if LR=='0.0':
                        LRval.append('EM')
                    else:
                        LRval.append('PyTorch, LR='+str(LR))
                    if init == 'unif':
                        initval.append('Rand. unif.')
                    elif init == '++':
                        initval.append('DC++')
                    elif init == 'dc':
                        initval.append('DC')
                    testtrainval.append(traintest)
                    modelorder.append(K)
            dftmp = pd.DataFrame({'Log likelihood': testlike, 'Optimizer': LRval, 'Initialization': initval,'Test or train':testtrainval,'Model order (K)':modelorder})
            df = pd.concat([df,dftmp])
  
sns.lineplot(ax=axs[0],x='Model order (K)',y='Log likelihood',hue='Initialization',style='Optimizer',data=df[df['Test or train']=='Train'])
sns.lineplot(ax=axs[1],x='Model order (K)',y='Log likelihood',hue='Initialization',style='Optimizer',data=df[df['Test or train']=='Test'])
plt.show()

plt.savefig('reports/figures/'+exptype+'_results_Watson_'+str(tt))
# fig.suptitle(name+' mixture model')
plt.show()
stop=7