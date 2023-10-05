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
LRs = [0,0.001,0.01,0.1,1]
exptype = 'synth'
for m in range(3):
    for tt in range(2): # train test likelihood
        fig, axs = plt.subplots(3, 3, figsize=(16, 9))
        if m==0:
            modelname='Watson'
        elif m==1:
            modelname='ACG'
        elif m==2:
            modelname = 'MACG'
        # left_labels = ['Number of components, K=2','Number of components, K=5','Number of components, K=10']
        for idx2,p in enumerate([3,10,25]):
            for idx1,K in enumerate([2,5,10]):
                if K>=p:
                    axs[idx1,idx2].set_visible(False)
                    continue
                df = pd.DataFrame(columns=['Log likelihood','model','Optimizer','Initialization'])
                for init in inits:
                    for LRidx,LR in enumerate(LRs):
                        if LR==1 or LR==0:
                            LRname = str(LR)+'.0'
                        else:
                            LRname = str(LR)
                        expname = '3d_'+init+'_'+LRname+'_p'+str(p)+'_K'+str(K)
                        file_path = 'data/results/'+exptype+'_outputs/'+modelname+'_'+expname+'_traintestlikelihood.csv'
                        try:
                            likes = np.loadtxt(file_path)
                        except:
                            continue

                        testlike = []
                        modelnames = []
                        LRval = []
                        initval = []
                        for rep in range(num_repl_outer):
                            
                            like = likes[tt,rep]
                            if like==0:
                                like=np.nan
                            testlike.append(like)
                            modelnames.append(modelname)
                            if LR==0:
                                LRval.append('EM')
                            else:
                                LRval.append('PyTorch, LR='+str(LR))
                            if init == 'unif':
                                initval.append('Rand. unif.')
                            elif init == '++':
                                initval.append('DC++')
                            elif init == 'dc':
                                initval.append('DC')
                        dftmp = pd.DataFrame({'Log likelihood': testlike, 'model': modelnames, 'Optimizer': LRval, 'Initialization': initval})
                        df = pd.concat([df,dftmp])
                try:
                    sns.violinplot(ax=axs[idx1,idx2],data=df[df.model==modelname],x='Initialization',y='Log likelihood',hue='Optimizer',inner='point',scale='count')
                except:
                    continue
                
                if idx1!=0 or idx2!=0:
                    if m==0 and tt==0:
                        axs[idx1,idx2].legend([],[], frameon=False)
                    
                # axs[idx1,idx2].grid(False)
                if tt == 0:
                    axs[idx1,idx2].set_title(modelname+' synthetic train log-likelihood, p='+str(p)+' K='+str(K), fontsize=12)
                elif tt == 1:
                    axs[idx1,idx2].set_title(modelname+' synthetic test log-likelihood, p='+str(p)+' K='+str(K), fontsize=12)
                axs[idx1,idx2].set_xlabel('')
                axs[idx1,idx2].get_yaxis().set_major_formatter(
                    ticker.FuncFormatter(lambda x, p: format(x)))
                
        # Add labels for the entire figure
        # fig.text(0.5, 0.04, 'X-axis Label', ha='center', fontsize=14)
        # fig.text(0.04, 0.5, 'Y-axis Label', va='center', rotation='vertical', fontsize=14)

        # Adjust the spacing between subplots
        fig.tight_layout()
        plt.savefig('reports/figures/synth_results_'+modelname+'_'+str(tt),dpi=600)
# fig.suptitle(name+' mixture model')
plt.show()
stop=7
