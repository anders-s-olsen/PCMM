import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
sns.set()

num_repl_outer = 10
inits = ['unif','++','dc']
LRs = [0,0.001,0.01,0.1]
exptype = '454'
for m in range(2):
    # if m==0:
    #     continue
    fig, axs = plt.subplots(3, figsize=(20, 10))
    if m==0:
        name='Watson'
    elif m==1:
        name='ACG'
    # left_labels = ['Number of components, K=2','Number of components, K=5','Number of components, K=10']
    for idx1,K in enumerate([2,5,10]):
        df = pd.DataFrame(columns=['Log likelihood','model','Optimizer','Initialization'])
        for init in inits:
            for LR in LRs:
                if LR==1 or LR==0:
                    LRname = str(LR)+'.0'
                else:
                    LRname = str(LR)
                expname = '3d_'+init+'_'+LRname+'_p454_K'+str(K)
                
                testlike = []
                modelname = []
                LRval = []
                initval = []
                for rep in range(num_repl_outer):
                    file_path = 'experiments/'+exptype+'_outputs/'+name+'_'+expname+'_traintestlikelihood_r'+str(rep)+'.csv'
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
                        initval.append('Rand. unif.')
                    elif init == '++':
                        initval.append('DC++')
                    elif init == 'dc':
                        initval.append('DC')
                dftmp = pd.DataFrame({'Log likelihood': testlike, 'model': modelname, 'Optimizer': LRval, 'Initialization': initval})
                df = pd.concat([df,dftmp])
        sns.violinplot(ax=axs[idx1],data=df[df.model==name],x='Initialization',y='Log likelihood',hue='Optimizer',inner='point',scale='count')
        axs[idx1].legend([],[], frameon=False)
        axs[idx1].set_title('Log-likelihood, K='+str(K), fontsize=12)
        axs[idx1].get_yaxis().set_major_formatter(
            ticker.FuncFormatter(lambda x, p: format(x)))
    # Add labels for the entire figure
    # fig.text(0.5, 0.04, 'X-axis Label', ha='center', fontsize=14)
    # fig.text(0.04, 0.5, 'Y-axis Label', va='center', rotation='vertical', fontsize=14)

    # Adjust the spacing between subplots
    fig.tight_layout()
# fig.suptitle(name+' mixture model')
plt.show()
stop=7