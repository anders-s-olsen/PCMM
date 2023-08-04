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
LRs = ['0.01','0.1','0.0']
exptype = '454'
for m in range(3):
    if m==1 or m==2:
        continue
    for tt in range(2): # train test likelihood
        fig, axs = plt.subplots(1, 3, figsize=(16, 3))
        if m==0:
            name='Watson'
        elif m==1:
            name='ACG'
        elif m==2:
            name = 'MACG'
        # left_labels = ['Number of components, K=2','Number of components, K=5','Number of components, K=10']
        for idx1,K in enumerate([2,5,10]):
            df = pd.DataFrame(columns=['Log likelihood','model','Optimizer','Initialization'])
            for init in inits:
                for LRidx,LR in enumerate(LRs):
                    expname = '454_'+init+'_'+LR+'_p454_K'+str(K)
                    
                    testlike = []
                    modelname = []
                    LRval = []
                    initval = []
                    for rep in range(num_repl_outer):
                        file_path = 'experiments/'+exptype+'_outputs/'+name+'_'+expname+'_traintestlikelihood_r'+str(rep)+'.csv'
                        try:
                            like = np.loadtxt(file_path)[tt]
                            testlike.append(like)
                        except: continue
                        modelname.append(name)
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
                    dftmp = pd.DataFrame({'Log likelihood': testlike, 'model': modelname, 'Optimizer': LRval, 'Initialization': initval})
                    df = pd.concat([df,dftmp])
            sns.violinplot(ax=axs[idx1],data=df[df.model==name],x='Initialization',y='Log likelihood',hue='Optimizer',inner='point',scale='count')
            if not(idx1==0):
                axs[idx1].legend([],[], frameon=False)
            # axs[idx1,idx2].grid(False)
            if tt == 0:
                axs[idx1].set_title(name+' 454-dimensional data training log-likelihood, K='+str(K), fontsize=12)
            elif tt == 1:
                axs[idx1].set_title(name+' 454-dimensional data test log-likelihood, K='+str(K), fontsize=12)
            axs[idx1].set_xlabel('')
            axs[idx1].get_yaxis().set_major_formatter(
                ticker.FuncFormatter(lambda x, p: format(x)))
                
        # Add labels for the entire figure
        # fig.text(0.5, 0.04, 'X-axis Label', ha='center', fontsize=14)
        # fig.text(0.04, 0.5, 'Y-axis Label', va='center', rotation='vertical', fontsize=14)

        # Adjust the spacing between subplots
        fig.tight_layout()
        plt.savefig('reports/figures/'+exptype+'_results_'+name+'_'+str(tt))
# fig.suptitle(name+' mixture model')
plt.show()
stop=7