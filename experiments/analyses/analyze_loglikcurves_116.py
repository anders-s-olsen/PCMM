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
inits = ['unif','++']
LRs = ['0.01','0.1','0.0','1.0']
modelnames = ['Watson','ACG','MACG']
for modelname in modelnames:
    fix,axs = plt.subplots(2,3,figsize=(14,11))
    for GSR in range(2):
        if GSR==0:
            exptype = '116'
        else:
            exptype = '116GSR'
        df = pd.DataFrame(columns=['Log likelihood','Optimizer','Initialization','Test or train','Model order (K)'])
        for idx1,K in enumerate(np.arange(2,31)):
            for init in inits:
                for LRidx,LR in enumerate(LRs):
                    expname = '116_'+init+'_'+LR+'_p116_K'+str(K)
                    testlike = []
                    LRval = []
                    initval = []
                    testtrainval = []
                    modelorder = []
                    file_path = 'experiments/'+exptype+'_outputs/'+modelname+'_'+expname+'_traintestlikelihood'+'.csv'
                    for tt,traintest in enumerate(['Train','Test','Test2']):
                        try:
                            like = np.loadtxt(file_path)[tt]
                            like = like[like!=0]
                            testlike.extend(like.tolist())
                        except: continue
                        for i in np.arange(len(like)):
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
        
        sns.lineplot(ax=axs[GSR,0],x='Model order (K)',y='Log likelihood',hue='Initialization',style='Optimizer',data=df[df['Test or train']=='Train'])
        sns.lineplot(ax=axs[GSR,1],x='Model order (K)',y='Log likelihood',hue='Initialization',style='Optimizer',data=df[df['Test or train']=='Test'])
        sns.lineplot(ax=axs[GSR,2],x='Model order (K)',y='Log likelihood',hue='Initialization',style='Optimizer',data=df[df['Test or train']=='Test2'])
        if GSR==0:
            axs[GSR,0].set_title('Train log likelihood')
            axs[GSR,1].set_title('Within-subject test log likelihood')
            axs[GSR,2].set_title('Between-subject test log likelihood')
        elif GSR==1:
            axs[GSR,0].set_title('GSR: Train log likelihood')
            axs[GSR,1].set_title('GSR: Within-subject test log likelihood')
            axs[GSR,2].set_title('GSR: Between-subject test log likelihood')

plt.show()


plt.savefig('reports/figures/'+exptype+'_results_Watson_'+str(tt))
# fig.suptitle(name+' mixture model')
plt.show()
stop=7