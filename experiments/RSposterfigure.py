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
init = 'unif'
LR = '0.0'
modelnames = ['Watson','ACG']
df = pd.DataFrame(columns=['Modelname','Log likelihood','Optimizer','Initialization','Test or train','Model order (K)'])
for modelname in modelnames:
    exptype = '116'
    for idx1,K in enumerate(np.arange(2,31)):
        expname = '116_'+init+'_'+LR+'_p116_K'+str(K)
        testlike = []
        LRval = []
        initval = []
        testtrainval = []
        modelorder = []
        file_path = 'experiments/'+exptype+'_outputs/'+modelname+'_'+expname+'_traintestlikelihood'+'.csv'
        for tt,traintest in enumerate(['Train','Test']):
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
        dftmp = pd.DataFrame({'Modelname':modelname,'Log likelihood': testlike, 'Optimizer': LRval, 'Initialization': initval,'Test or train':testtrainval,'Model order (K)':modelorder})
        df = pd.concat([df,dftmp])
        
plt.figure(figsize=(4,3))
sns.lineplot(x='Model order (K)',y='Log likelihood',style='Test or train',hue='Modelname',data=df)
plt.show()
ax = plt.gca()
ax.legend(
    labels = ['Watson train', '', 'ACG train', 'ACG test'],
    fontsize='large',
    title_fontsize='x-large')
sns.lineplot(x='Model order (K)',y='Log likelihood',style='Test or train',data=df[df['Modelname']=='Watson'],legend=False,color='b')

ax.set_ylabel('Watson log likelihood', color='b')
ax.tick_params(axis='y', labelcolor='b')
ax.set_ylim([-1000,0])
ax2 = plt.twinx()
sns.lineplot(x='Model order (K)',y='Log likelihood',style='Test or train',data=df[df['Modelname']=='ACG'],legend=False,color='r', ax=ax2)
ax2.set_ylabel('ACG log likelihood', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylim([-1000,0])
ax2.set_title('Train log likelihood')
ax2.legend(['ACG','Watson'])
plt.show()


# Create the first axis (left)


plt.show()


# plt.savefig('reports/figures/'+exptype+'_results_Watson_'+str(tt))
stop=7