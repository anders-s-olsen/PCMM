# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs('experiments_phaserando/figures',exist_ok=True)
def rename(df):
    #rename column LR to Inference
    df = df.rename(columns={'LR': 'Inference'})
    #rename Inference level 0 to 'EM'
    df['Inference'] = df['Inference'].replace(0, 'EM')
    df['Inference'] = df['Inference'].replace(0.1, 'PyTorch')
    # go through each row in the dataframe and add a name column
    names = []
    for i in range(len(df)):
        if 'Watson' in df['modelname'][i]:
            if df['HMM'][i]==False:
                names.append(df['modelname'][i])
            else:
                names.append(df['modelname'][i]+' HMM')
        elif df['modelname'][i] in ['ACG','MACG']:
            if df['HMM'][i]==False:
                names.append(df['modelname'][i]+' rank='+str(df['ACG_rank'][i]))
            else:
                names.append(df['modelname'][i]+' rank='+str(df['ACG_rank'][i])+' HMM')
        else:
            names.append('Unknown')
    df['names'] = names
    return df

data_folder = 'data/results/torchvsEM_phase_controlled_results/'
df1 = pd.read_csv(data_folder+'phase_controlled_EM_Watson.csv')
df2 = pd.read_csv(data_folder+'phase_controlled_EM_ACG.csv')
df3 = pd.read_csv(data_folder+'phase_controlled_EM_MACG.csv')
df4 = pd.read_csv(data_folder+'phase_controlled_torch_Watson.csv')
df5 = pd.read_csv(data_folder+'phase_controlled_torch_ACG.csv')
df6 = pd.read_csv(data_folder+'phase_controlled_torch_MACG.csv')
df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
df = rename(df)
df1 = pd.read_csv(data_folder+'phase_controlled_EM_Watson_initprevious.csv')
df2 = pd.read_csv(data_folder+'phase_controlled_EM_ACG_initprevious.csv')
df3 = pd.read_csv(data_folder+'phase_controlled_EM_MACG_initprevious.csv')
df4 = pd.read_csv(data_folder+'phase_controlled_torch_Watson_initprevious.csv')
df5 = pd.read_csv(data_folder+'phase_controlled_torch_ACG_initprevious.csv')
df6 = pd.read_csv(data_folder+'phase_controlled_torch_MACG_initprevious.csv')
df_initprevious = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
df_initprevious = rename(df_initprevious)
df1 = pd.read_csv(data_folder+'phase_controlled_EM_Watson_initEM.csv')
df2 = pd.read_csv(data_folder+'phase_controlled_EM_ACG_initEM.csv')
df3 = pd.read_csv(data_folder+'phase_controlled_EM_MACG_initEM.csv')
df4 = pd.read_csv(data_folder+'phase_controlled_torch_Watson_initEM.csv')
df5 = pd.read_csv(data_folder+'phase_controlled_torch_ACG_initEM.csv')
df6 = pd.read_csv(data_folder+'phase_controlled_torch_MACG_initEM.csv')
df_initEM = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
df_initEM = rename(df_initEM)

order = ['Watson','Watson HMM', 
         'ACG rank=1','ACG rank=1 HMM', 'ACG rank=5','ACG rank=5 HMM', 'ACG rank=10','ACG rank=10 HMM', 'ACG rank=25','ACG rank=25 HMM', 'ACG rank=50','ACG rank=50 HMM','ACG rank=fullrank', 
         'MACG rank=1','MACG rank=1 HMM', 'MACG rank=5','MACG rank=5 HMM', 'MACG rank=10','MACG rank=10 HMM', 'MACG rank=25','MACG rank=25 HMM', 'MACG rank=50','MACG rank=50 HMM','MACG rank=fullrank']
plt.figure(figsize=(20,3))
sns.boxplot(x='names', y='train_NMI', data=df,hue='Inference', order=order)
plt.title('Initialization with dc++_seg')
plt.xticks(rotation=45)
plt.ylim(0,1)
plt.savefig('experiments_phaserando/figures/phase_controlled_results_initdcseg.png',bbox_inches='tight')

order = ['Watson','Watson HMM', 
         'ACG rank=1','ACG rank=1 HMM', 'ACG rank=5','ACG rank=5 HMM', 'ACG rank=10','ACG rank=10 HMM', 'ACG rank=25','ACG rank=25 HMM', 'ACG rank=50','ACG rank=50 HMM','ACG rank=fullrank', 
         'MACG rank=1','MACG rank=1 HMM', 'MACG rank=5','MACG rank=5 HMM', 'MACG rank=10','MACG rank=10 HMM', 'MACG rank=25','MACG rank=25 HMM', 'MACG rank=50','MACG rank=50 HMM','MACG rank=fullrank']
plt.figure(figsize=(20,3))
sns.boxplot(x='names', y='train_NMI', data=df_initprevious,hue='Inference', order=order)
plt.title('Initialization with previous model')
plt.xticks(rotation=45)
plt.ylim(0,1)
plt.savefig('experiments_phaserando/figures/phase_controlled_results_initprevious.png',bbox_inches='tight')

order = ['Watson','Watson HMM', 
         'ACG rank=1','ACG rank=1 HMM', 'ACG rank=5','ACG rank=5 HMM', 'ACG rank=10','ACG rank=10 HMM', 'ACG rank=25','ACG rank=25 HMM', 'ACG rank=50','ACG rank=50 HMM','ACG rank=fullrank', 
         'MACG rank=1','MACG rank=1 HMM', 'MACG rank=5','MACG rank=5 HMM', 'MACG rank=10','MACG rank=10 HMM', 'MACG rank=25','MACG rank=25 HMM', 'MACG rank=50','MACG rank=50 HMM','MACG rank=fullrank']
plt.figure(figsize=(20,3))
sns.boxplot(x='names', y='train_NMI', data=df_initEM,hue='Inference', order=order)
plt.title('Initialization with EM')
plt.xticks(rotation=45)
plt.ylim(0,1)
plt.savefig('experiments_phaserando/figures/phase_controlled_results_initEM.png',bbox_inches='tight')