clear
folder = '/dtu-compute/HCP_dFC/2023/hcp_dfc/src/visualization/';
cd(folder)
save = true;

data_raw = h5read([folder,'fits/cluster_data.h5'],'/data');
evs_raw = h5read([folder,'fits/cluster_data.h5'],'/evs');
data = [squeeze(data_raw(1,:,:)),squeeze(data_raw(2,:,:))];

viewangles = [-60,30;30,30];

target = zeros(3,3,2);
target(:,:,1) = [1,0,-1;0,1,0;-1,0,1];
target(:,:,2) = [1,-1,0;-1,1,0;0,0,1];

%% actual data (3D)
close all
col1 = [0, 0.9, 0];
col2 = [0, 0.5, 0];
col3 = [0.9, 0, 0];
col4 = [0.5, 0, 0];
cols = [col1; col2; col3; col4];

seq = [ones(1000,1);3*ones(1000,1);2*ones(1000,1);4*ones(1000,1)];
data2 = data.*(sqrt([evs_raw(1,:),evs_raw(2,:)]));

pointsaxisfig(data2',seq,cols,viewangles(1,:),[],false,target)
if save;exportgraphics(gcf,[folder,'figs/datafigaxis_view1.png'],'resolution',300);end
pointsaxisfig(data2',seq,cols,viewangles(2,:),[],false,target)
if save;exportgraphics(gcf,[folder,'figs/datafigaxis_view2.png'],'resolution',300);end

%% actual data (sphere)
close all
col1 = [0, 0.9, 0];
col2 = [0, 0.5, 0];
col3 = [0.9, 0, 0];
col4 = [0.5, 0, 0];
cols = [col1; col2; col3; col4];

seq = [ones(1000,1);3*ones(1000,1);2*ones(1000,1);4*ones(1000,1)];

pointsspherefig(data',seq,cols,viewangles(1,:),[],false,target)
if save;exportgraphics(gcf,[folder,'figs/datafig_view1.png'],'resolution',300);end
pointsspherefig(data',seq,cols,viewangles(2,:),[],false,target)
if save;exportgraphics(gcf,[folder,'figs/datafig_view2.png'],'resolution',300);end

%% kmeans fit

data_flip = data;
data_flip(:,sum(data_flip>0,1)>1) = -data_flip(:,sum(data_flip>0,1)>1);

cols = [0,0.5,0.5;0.5,0,0.5];

label = table2array(readtable([folder,'/fits/kmeans_labels.txt']));
centroid = table2array(readtable([folder,'/fits/kmeans_centroids.txt']))';
seq = zeros(size(data,2)/2,1);
if label(1,1)==1
seq(logical(label(1,:))) = 1;
seq(logical(label(2,:))) = 2;
else
seq(logical(label(1,:))) = 2;
seq(logical(label(2,:))) = 1;
centroid = [centroid(:,2),centroid(:,1)];
end
pointsspherefig(data_flip(:,1:2000)',seq,cols,viewangles(1,:),centroid,true,target)
if save;exportgraphics(gcf,[folder,'figs/kmeans_view1.png'],'resolution',300);end
pointsspherefig(data_flip(:,1:2000)',seq,cols,viewangles(2,:),centroid,true,target)
if save;exportgraphics(gcf,[folder,'figs/kmeans_view2.png'],'resolution',300);end

%% diametrical fit

cols = [0,0.5,0.5;0.5,0,0.5];

label = table2array(readtable([folder,'/fits/diametrical_labels.txt']));
centroid = table2array(readtable([folder,'/fits/diametrical_centroids.txt']));
seq = zeros(size(data,2)/2,1);
if label(1,1)==1
seq(logical(label(1,:))) = 1;
seq(logical(label(2,:))) = 2;
else
seq(logical(label(1,:))) = 2;
seq(logical(label(2,:))) = 1;
centroid = [centroid(:,2),centroid(:,1)];
end

pointsspherefig(data(:,1:2000)',seq,cols,viewangles(1,:),centroid,false,target)
if save;exportgraphics(gcf,[folder,'figs/diametrical_view1.png'],'resolution',300);end
pointsspherefig(data(:,1:2000)',seq,cols,viewangles(2,:),centroid,false,target)
if save;exportgraphics(gcf,[folder,'figs/diametrical_view2.png'],'resolution',300);end

%% grassmann fit

cols = [0,0.5,0.5;0.5,0,0.5];

label = table2array(readtable([folder,'/fits/grassmann_labels.txt']));
centroid1 = table2array(readtable([folder,'/fits/grassmann_centroids1.txt']));
centroid2 = table2array(readtable([folder,'/fits/grassmann_centroids2.txt']));
seq = zeros(size(data,2)/2,1);
centroids = zeros(2,3,2);
if label(1,1)==1
seq(logical(label(1,:))) = 1;
seq(logical(label(2,:))) = 2;
centroids(1,:,:) = centroid1;
centroids(2,:,:) = centroid2;
else
seq(logical(label(1,:))) = 2;
seq(logical(label(2,:))) = 1;
centroids(1,:,:) = centroid2;
centroids(2,:,:) = centroid1;
end
seq = [seq;seq];

pointsspherefig(data',seq,cols,viewangles(1,:),centroids,false,target)
if save;exportgraphics(gcf,[folder,'figs/grassmann_view1.png'],'resolution',300);end
pointsspherefig(data',seq,cols,viewangles(2,:),centroids,false,target)
if save;exportgraphics(gcf,[folder,'figs/grassmann_view2.png'],'resolution',300);end

%% weighted grassmann fit

cols = [0,0.5,0.5;0.5,0,0.5];

label = table2array(readtable([folder,'/fits/weighted_grassmann_labels.txt']));
centroid1 = table2array(readtable([folder,'/fits/weighted_grassmann_centroids1.txt']));
centroid1_weight = table2array(readtable([folder,'/fits/weighted_grassmann_centroids_weights1.txt']));
centroid2 = table2array(readtable([folder,'/fits/weighted_grassmann_centroids2.txt']));
centroid2_weight = table2array(readtable([folder,'/fits/weighted_grassmann_centroids_weights2.txt']));
seq = zeros(size(data,2)/2,1);
centroids = zeros(2,3,2);
if label(1,1)==1
seq(logical(label(1,:))) = 1;
seq(logical(label(2,:))) = 2;
centroids(1,:,:) = centroid1.*(sqrt(centroid1_weight'));
centroids(2,:,:) = centroid2.*(sqrt(centroid2_weight'));
else
seq(logical(label(1,:))) = 2;
seq(logical(label(2,:))) = 1;
centroids(1,:,:) = centroid2.*(sqrt(centroid2_weight'));
centroids(2,:,:) = centroid1.*(sqrt(centroid1_weight'));
end
seq = [seq;seq];

pointsaxisfig(data2',seq,cols,viewangles(1,:),centroids,false,target)
if save;exportgraphics(gcf,[folder,'figs/weighted_grassmann_view1.png'],'resolution',300);end
pointsaxisfig(data2',seq,cols,viewangles(2,:),centroids,false,target)
if save;exportgraphics(gcf,[folder,'figs/weighted_grassmann_view2.png'],'resolution',300);end

%% Watson MM fit
close all
mu = table2array(readtable([folder,'/fits/watson_centroids_mu.txt']));
kappa = table2array(readtable([folder,'/fits/watson_centroids_kappa.txt']));
pi = table2array(readtable([folder,'/fits/watson_centroids_pi.txt']));

params = struct;
params.mu = mu;
params.kappa = kappa;
params.pi = pi;

contourspherefig('Watson',params,viewangles(1,:),0.75,target,'type1')
if save;exportgraphics(gcf,[folder,'figs/Watson_view1.png'],'resolution',300);end
contourspherefig('Watson',params,viewangles(2,:),0.75,target,'type1')
if save;exportgraphics(gcf,[folder,'figs/Watson_view2.png'],'resolution',300);end

%% ACG MM fit
% close all
L1 = table2array(readtable([folder,'/fits/ACG_centroids1.txt']));
L2 = table2array(readtable([folder,'/fits/ACG_centroids2.txt']));
L = zeros(3,3,2);
L(:,:,1) = L1;
L(:,:,2) = L2;
% pi = table2array(readtable([folder,'/fits/ACG_centroids_pi.txt']));

params = struct;
params.L = L;
% params.pi = pi;

contourspherefig('ACG',params,viewangles(1,:),0.75,target,'type1')
% contourspherefig('ACG',params,viewangles(1,:),0.85,target,'type2')
if save;exportgraphics(gcf,[folder,'figs/ACG_view1.png'],'resolution',300);end
contourspherefig('ACG',params,viewangles(2,:),0.75,target,'type1')
if save;exportgraphics(gcf,[folder,'figs/ACG_view2.png'],'resolution',300);end

%% MACG MM fit
% close all
L1 = table2array(readtable([folder,'/fits/MACG_centroids1.txt']));
L2 = table2array(readtable([folder,'/fits/MACG_centroids2.txt']));
L = zeros(3,3,2);
L(:,:,1) = L1;
L(:,:,2) = L2;
% pi = table2array(readtable([folder,'/fits/ACG_centroids_pi.txt']));

params = struct;
params.L = L;
% params.pi = pi;
contourspherefig('MACG',params,viewangles(1,:),0.75,target,'type1')
% contourspherefig('MACG',params,viewangles(1,:),0.85,target,'type2')
if save;exportgraphics(gcf,[folder,'figs/MACG_view1.png'],'resolution',300);end
contourspherefig('MACG',params,viewangles(2,:),0.75,target,'type1')
if save;exportgraphics(gcf,[folder,'figs/MACG_view2.png'],'resolution',300);end
