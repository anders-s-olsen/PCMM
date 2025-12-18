import numpy as np
import pandas as pd
from paper.helper_functions_paper import calc_NMI

modelnames = ['Complex_ACG','Normal','Complex_Normal','least_squares','diametrical','complex_diametrical']
# load posteriors

for experiment in ['all_tasks']: #, REST1REST2, all_tasks
    for dataset in ['fMRI_SchaeferTian116', 'fMRI_SchaeferTian116_GSR', 'fMRI_SchaeferTian232_GSR', 'fMRI_Schaefer400_GSR', '2025fMRI_SchaeferTian116_GSR']:# fMRI_SchaeferTian116, fMRI_SchaeferTian116_GSR, fMRI_SchaeferTian232_GSR, fMRI_Schaefer400_GSR
        if experiment == 'REST1REST2' and dataset!='fMRI_SchaeferTian116_GSR':
            continue
        for modelname in modelnames:
            aggregated_df = pd.DataFrame()
            computational_reproducibility_df = pd.DataFrame()
            if experiment=='all_tasks':
                K_range = range(7,8)
            else:
                K_range = range(1,11)
            for K in K_range:
                print('Processing experiment',experiment,'dataset',dataset,'modelname',modelname,'K',K)
                for rank in [1,5,10,25,50,100]:
                    if modelname in ['least_squares','diametrical','complex_diametrical'] and rank!=25:
                        continue
                    posteriors_train = []
                    posteriors_test = []
                    for inner in range(10):
                        try:
                            df = pd.read_csv('paper/data/results/'+dataset+'/dfs/'+experiment+'modelorder_realdata_'+modelname+'_K='+str(K)+'_rank='+str(rank)+'_inner='+str(inner)+'.csv')
                            df2 = df.loc[(df['modelname']==modelname) & (df['K']==K) & (df['rank']==rank) & (df['inner']==inner)].copy()
                            if K>1 and experiment == 'REST1REST2':
                                posterior_train = np.loadtxt('paper/data/results/'+dataset+'/posteriors/'+experiment+'modelorder_realdata_'+modelname+'_K='+str(K)+'_rank='+str(rank)+'_inner='+str(inner)+'_train.txt',delimiter=',')
                                posterior_test = np.loadtxt('paper/data/results/'+dataset+'/posteriors/'+experiment+'modelorder_realdata_'+modelname+'_K='+str(K)+'_rank='+str(rank)+'_inner='+str(inner)+'_test.txt',delimiter=',')
                                print('Processing posterior for modelname',modelname,'K',K, 'rank',rank, 'inner',inner)
                                posteriors_train.append(posterior_train)
                                posteriors_test.append(posterior_test)

                            # else:
                            #     print('Skipping entropy calculation for K=1 for modelname',modelname,'K',K, 'rank',rank)
                        except:
                            print('No df or posterior file for modelname',modelname,'K',K, 'rank',rank, 'inner',inner)
                            continue
                        aggregated_df = pd.concat([aggregated_df, df2], ignore_index=True)
                    
                    if experiment == 'REST1REST2':
                        if K==1:
                            df_tmp = pd.DataFrame({'modelname':[modelname],'K':[K],'rank':[rank],'dataset':[dataset],'inner':[0],'inner2':[0],'NMI_train':[1.0], 'NMI_test':[1.0]})
                            computational_reproducibility_df = pd.concat([computational_reproducibility_df, df_tmp], ignore_index=True)
                            continue
                        for inner in range(10):
                            for inner2 in range(inner+1,10):
                                try:
                                    NMI_train = calc_NMI(posteriors_train[inner], posteriors_train[inner2])
                                    NMI_test = calc_NMI(posteriors_test[inner], posteriors_test[inner2])
                                    df_tmp = pd.DataFrame({'modelname':[modelname],'K':[K],'rank':[rank],'dataset':[dataset],'inner':[inner],'inner2':[inner2],'NMI_train':[NMI_train],'NMI_test':[NMI_test]})
                                    computational_reproducibility_df = pd.concat([computational_reproducibility_df, df_tmp], ignore_index=True)
                                except:
                                    continue
                if experiment == 'REST1REST2':
                    computational_reproducibility_df.to_csv('paper/data/results/'+dataset+'/'+experiment+'_computational_reproducibility_modelorder_realdata_'+modelname+'.csv', index=False)
                aggregated_df.to_csv('paper/data/results/'+dataset+'/'+experiment+'_aggregated_df_modelorder_realdata_'+modelname+'.csv', index=False)
                
