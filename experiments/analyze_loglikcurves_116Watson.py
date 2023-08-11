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
    fix,axs = plt.subplots(2,3,figsize=(16,9))
    for GSR in range(2):
        if GSR==0:
            exptype = '116'
        else:
            exptype = '116GSR'
        df = pd.DataFrame(columns=['Log likelihood','Optimizer','Initialization','Test or train','Model order (K)'])
        for idx1,K in enumerate(np.arange(2,31)):
            for init in inits:
                for LRidx,LR in enumerate(LRs):
                    if modelname != 'MACG':
                        expname = '116_'+init+'_'+LR+'_p116_K'+str(K)
                    else:
                        expname = '116_'+init+'_'+LR+'_p2_K'+str(K)
                    testlike = []
                    LRval = []
                    initval = []
                    testtrainval = []
                    modelorder = []
                    for rep in range(num_repl_outer):
                        for tt,traintest in enumerate(['Train','Test','Test2']):
                            file_path = 'experiments/'+exptype+'_outputs/'+modelname+'_'+expname+'_traintestlikelihood_r'+str(rep)+'.csv'
                            try:
                                like = np.loadtxt(file_path)[tt][rep]
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
        
        sns.lineplot(ax=axs[GSR,0],x='Model order (K)',y='Log likelihood',hue='Initialization',style='Optimizer',data=df[df['Test or train']=='Train'])
        sns.lineplot(ax=axs[GSR,1],x='Model order (K)',y='Log likelihood',hue='Initialization',style='Optimizer',data=df[df['Test or train']=='Test'])
        sns.lineplot(ax=axs[GSR,2],x='Model order (K)',y='Log likelihood',hue='Initialization',style='Optimizer',data=df[df['Test or train']=='Test2'])
plt.show()

plt.savefig('reports/figures/'+exptype+'_results_Watson_'+str(tt))
# fig.suptitle(name+' mixture model')
plt.show()
stop=7