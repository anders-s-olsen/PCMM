import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
sns.set()
sns.set_theme(style="whitegrid")
# matplotlib.rcParams.update({'font.size': 10})

num_repl_outer = 10
init = '++'
LR = '0.1'
exptype = '454'
modelnames = ['ACG','MACG']

for m,modelname in enumerate(modelnames):
    fix,axs = plt.subplots(1,3,figsize=(16,9))
    if modelname=='MACG':
        continue
    # left_labels = ['Number of components, K=2','Number of components, K=5','Number of components, K=10']
    df = pd.DataFrame(columns=['Log likelihood','Model','Rank','Model order (K)','Test or Train'])
    likevals = []
    model = []
    rankval = []
    Kval = []
    testtrainval = []
    for idx1,K in enumerate(np.arange(2,11)):
        expname = '454_full_'+init+'_'+str(LR)+'_p454_K'+str(K)
        
        for ttidx,tt in enumerate(['Train','Test']): # train test likelihood
            for rep in range(num_repl_outer):
                for rank in np.arange(1,200,2):
                    file_path = 'experiments/'+exptype+'_outputs/'+modelname+'_'+expname+'_traintestlikelihood_r'+str(rep)+'_rank'+str(rank)+'.csv'
                    try:
                        like = np.loadtxt(file_path)[ttidx]
                        likevals.append(like)
                    except: continue
                    model.append(modelname)
                    rankval.append(rank)
                    Kval.append(K)
                    testtrainval.append(tt)
        dftmp = pd.DataFrame({'Log likelihood': likevals, 'Model': model, 'Rank': rankval,'Model order (K)':Kval,'Test or Train':testtrainval})
        df = pd.concat([df,dftmp])

        # sns.lineplot(ax=axs[m],x='Model order (K)',y='Log likelihood',hue='Test or Train',data=df)
    # plt.figure()
    # sns.relplot(x='Model order (K)',y='Rank',size='Log likelihood',hue='Log likelihood',palette='vlag',data=df[df['Test or Train']=='Test'])
    sns.lineplot(ax=axs[0],x='Rank',y='Log likelihood',hue='Model order (K)',data=df[df['Test or Train']=='Train'])
    sns.lineplot(ax=axs[1],x='Rank',y='Log likelihood',hue='Model order (K)',data=df[df['Test or Train']=='Test'])
    # sns.lineplot(ax=axs[0],x='rank',y='Log likelihood',size='Log likelihood',hue='Model order (K)',data=df[df['Test or Train']=='Test2'])

    # plt.savefig('reports/figures/'+exptype+'_results_'+name+'_'+str(tt))
# fig.suptitle(name+' mixture model')
plt.show()
stop=7