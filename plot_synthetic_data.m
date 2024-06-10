close all
addpath(genpath('src'))
X = h5read('data/synthetic/phase_controlled_3data_eida.h5','/U');

seq = repmat([ones(600,1);2*ones(600,1)],10,1);

pointsspherefig(squeeze(X(1,:,:))',seq)
pointsspherefig(squeeze(X(2,:,:))',seq)
% pointsspherefig([squeeze(X(1,:,:)),squeeze(X(2,:,:))]',repelem(seq,2))