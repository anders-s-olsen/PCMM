clear,close all
addpath(genpath(pwd))

p=3;
sig2 = eye(3)+0.99*(ones(3)-eye(3)); %noise is one minus the off diagonal element, log space
sig2 = p*sig2/trace(sig2);
sig3 = diag([1e-2,1,1])+0.9*[0,0,0;0,0,1;0,1,0]; %noise is the first diagonal element, log space
sig3 = p*sig3/trace(sig3);
SIGMAs = cat(3,sig2,sig3);

%% Figure 1

X=table2array(readtable('data/synthetic/synth_data_ACG.csv'));
cluster_id = [ones(500,1);2*ones(500,1)];
pointsspherefig(X,cluster_id);

%% Figure 2

rep = '2';
lr = '0.1';
init = '++';
foldertotakefrom = ['experiments/outputs3d_',init,'_',lr,'/'];

%%% Watson MM
% WMM_results = WMM_EM_BigMem2(X,2,200,1,'++',0);mu1 = WMM_results.mu(:,1);mu2 = WMM_results.mu(:,2);
mu = table2array(readtable([foldertotakefrom,'Watson_3d_',init,'_',lr,'_mu_K=2_r',rep,'.csv']));
kappa = table2array(readtable([foldertotakefrom,'Watson_3d_',init,'_',lr,'_kappa_K=2_r',rep,'.csv']));
orderwmm = contourspherefig(mu,kappa,[],SIGMAs);
% pause(2)
% exportgraphics(gcf,[ff,'sphere_WMM_contour.png'],'Resolution',300)

%%% ACG MM
L1 = table2array(readtable([foldertotakefrom,'ACG_3d_',init,'_',lr,'_L_K=2_k0_r',rep,'.csv']));
L2 = table2array(readtable([foldertotakefrom,'ACG_3d_',init,'_',lr,'_L_K=2_k1_r',rep,'.csv']));
orderacgmm = contourspherefig([],[],cat(3,L1,L2),SIGMAs);
% pause(2)
% exportgraphics(gcf,[ff,'sphere_ACGMM_contour.png'],'Resolution',300)
