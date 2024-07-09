# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
def rename(df):
    #rename grassmann to Grassmann
    df['modelname'] = df['modelname'].replace('euclidean','Euclidean K-means, sign-fliped')
    df['modelname'] = df['modelname'].replace('diametrical','Diametrical clustering')
    df['modelname'] = df['modelname'].replace('grassmann','Grassmann clustering')
    df['modelname'] = df['modelname'].replace('weighted_grassmann','Weighted Grassmann clustering')
    df['modelname'] = df['modelname'].replace('SingularWishart','Singular Wishart')
    
    names = []
    for i in range(len(df)):
        if df['modelname'][i] in ['ACG','MACG','Singular Wishart']:
            names.append(df['modelname'][i]+' rank='+str(int(df['rank'][i])))
        else:
            names.append(df['modelname'][i])
    df['names'] = names
    return df

# %%
data_folder = 'data/results/torchvsEM_phase_controlled_results/'
df1 = pd.read_csv(data_folder+'phase_controlled_EM_euclidean_initprevious.csv')
df2 = pd.read_csv(data_folder+'phase_controlled_EM_diametrical_initprevious.csv')
df3 = pd.read_csv(data_folder+'phase_controlled_EM_grassmann_initprevious.csv')
df4 = pd.read_csv(data_folder+'phase_controlled_EM_weighted_grassmann_initprevious.csv')
df5 = pd.read_csv(data_folder+'phase_controlled_EM_Watson_initprevious.csv')
df6 = pd.read_csv(data_folder+'phase_controlled_EM_ACG_initprevious.csv')
df7 = pd.read_csv(data_folder+'phase_controlled_EM_MACG_initprevious.csv')
df8 = pd.read_csv(data_folder+'phase_controlled_EM_SingularWishart_initprevious.csv')
df = pd.concat([df1, df2, df3, df4, df5, df6,df7,df8], ignore_index=True)
df = rename(df)


# %%
order = ['Euclidean K-means, sign-fliped','Diametrical clustering','Grassmann clustering','Weighted Grassmann clustering',
         'Watson','ACG rank=5','ACG rank=25',
         'MACG rank=5','MACG rank=25','Singular Wishart rank=5','Singular Wishart rank=25']

plt.figure(figsize=(20,3))
sns.boxplot(x='names', y='train_NMI', data=df, order=order)
plt.ylim(0,1)
plt.ylabel('Normalized mutual information')
plt.xlabel('')
# plt.title('Phase-controlled data results')
plt.xticks(rotation=45);
plt.savefig('phase_controlled_data_results.png',bbox_inches='tight',dpi=300)

# %%
# same as above but where labels are on the axis and NMI on the x-axis
plt.figure(figsize=(5,7))
sns.boxplot(x='train_NMI', y='names', data=df, order=order)
plt.xlim(0,1)
plt.xlabel('Normalized mutual information')
plt.ylabel('')
# plt.title('Phase-controlled data results')
plt.savefig('phase_controlled_data_results_transposed.png',bbox_inches='tight',dpi=300)


