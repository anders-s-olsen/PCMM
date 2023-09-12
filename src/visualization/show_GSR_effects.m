data_GSR = h5read('C:\Users\ansol\OneDrive\OneDrive - Danmarks Tekniske Universitet\Dokumenter\PhD\HCP\hcp_dfc\data\processed\fMRI_SchaeferTian116_GSR_RL1.h5','/Dataset',[1,1],[1200,116]);
data = h5read('C:\Users\ansol\OneDrive\OneDrive - Danmarks Tekniske Universitet\Dokumenter\PhD\HCP\hcp_dfc\data\processed\fMRI_SchaeferTian116_RL1.h5','/Dataset',[1,1],[1200,116]);

randidx = randsample(116,3);
pointsspherefig(data(:,randidx)./vecnorm(data(:,randidx),2,2),ones(size(data,1),1))
pointsspherefig(data_GSR(:,randidx)./vecnorm(data_GSR(:,randidx),2,2),ones(size(data,1),1))





