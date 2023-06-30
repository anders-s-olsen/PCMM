clear,close all

%% ACG mixture K=2, p=3
p=3;
sig2 = eye(3)+0.99*(ones(3)-eye(3)); %noise is one minus the off diagonal element, log space
sig2 = p*sig2/trace(sig2);
sig3 = diag([1e-2,1,1])+0.9*[0,0,0;0,0,1;0,1,0]; %noise is the first diagonal element, log space
sig3 = p*sig3/trace(sig3);
SIGMAs = cat(3,sig2,sig3);

% train data
[X,cluster_id] = syntheticACGMixture([zeros(1,500),ones(1,500);ones(1,500),zeros(1,500)]',SIGMAs,1000,0);
pointsspherefig(X,cluster_id);
writetable(array2table(X),'data/synthetic/synth_data_ACG.csv','WriteVariableNames',false)

% test data
[X,cluster_id] = syntheticACGMixture([zeros(1,500),ones(1,500);ones(1,500),zeros(1,500)]',SIGMAs,1000,0);
pointsspherefig(X,cluster_id);
writetable(array2table(X),'data/synthetic/synth_data_ACG2.csv','WriteVariableNames',false)

%%
[X,cluster_id] = syntheticMACGMixture([zeros(1,500),ones(1,500);ones(1,500),zeros(1,500)]',SIGMAs,1000,2,0);
pointsspherefig(X(:,:,1),cluster_id);
pointsspherefig(X(:,:,2),cluster_id);
X2 = zeros(2*size(X,1),3);
X2(1:2:2000,:) = X(:,:,1);
X2(2:2:2000,:) = X(:,:,2);
writetable(array2table(X2),'data/synthetic/synth_data_MACG.csv','WriteVariableNames',false)
return
%% generate data according to noise levels

noise = logspace(-3,0,7);
noisedB = 20*log10(noise);
noisedB

for i = 1:numel(noise)
    sig2 = eye(3)+(1-noise(i))*(ones(3)-eye(3)); %noise is one minus the off diagonal element, log space
    sig3 = diag([noise(i),1,1])+0.9*[0,0,0;0,0,1;0,1,0]; %noise is the first diagonal element, log space
    
    SIGMAs = cat(3,sig2,sig3);
%     [X,cluster_id] = syntheticMixture3D(PI,SIGMAs,size(PI,1),0);
    [X,cluster_id] = syntheticMixture3Dv2(PI,SIGMAs,size(PI,1),0);
        pointsspherefig(X,cluster_id);
    delete(['data/synthetic_noise/v2HMMdata_noise_',num2str(noisedB(i)),'.h5'])
    h5create(['data/synthetic_noise/v2HMMdata_noise_',num2str(noisedB(i)),'.h5'],'/X',size(X))
    h5write(['data/synthetic_noise/v2HMMdata_noise_',num2str(noisedB(i)),'.h5'],'/X',X)
    h5create(['data/synthetic_noise/v2HMMdata_noise_',num2str(noisedB(i)),'.h5'],'/cluster_id',size(cluster_id))
    h5write(['data/synthetic_noise/v2HMMdata_noise_',num2str(noisedB(i)),'.h5'],'/cluster_id',cluster_id)
    
end


%% ACG sampler
function [X,cluster_allocation] = syntheticACGMixture(PI,SIGMAs,num_points,noise)

num_clusters = size(SIGMAs,3);
point_dim = size(SIGMAs,2);

X = zeros(num_points,point_dim);
cluster_allocation = zeros(num_points,1);
for n = 1:num_points
    n_clust_id = randsample(num_clusters,1,true,PI(n,:));
    x_i = mvnrnd(zeros(point_dim,1),SIGMAs(:,:,n_clust_id),1);
    X(n,:) = x_i/norm(x_i);
    % nx = chol(SIGMAs(:,:,n_clust_id),'lower') * randn(point_dim,1)+noise*randn(point_dim,1);
    % X(n,:) = nx/norm(nx);
    cluster_allocation(n) = n_clust_id;
end
end

%% MACG sampler
function [X,cluster_allocation] = syntheticMACGMixture(PI,SIGMAs,num_points,num_cols,noise)

num_clusters = size(SIGMAs,3);
point_dim = size(SIGMAs,2);

target = [0,1;1,1;1,1];

X = zeros(num_points,point_dim,num_cols);
cluster_allocation = zeros(num_points,1);
for n = 1:num_points
    n_clust_id = randsample(num_clusters,1,true,PI(n,:));
    Xi = mvnrnd(zeros(point_dim,1),SIGMAs(:,:,n_clust_id),num_cols)';
    Xsq = (Xi'*Xi)^(-0.5);
    tmp = Xi*Xsq;
    % sort
    sim = (tmp'*(target./vecnorm(target))).^2;
    [val1,id1] = max(sim(1,:));
    [val2,id2] = max(sim(2,:));
    if id1==id2
        if val1>val2
            id2 = setdiff([1,2],id1);
        else
            id1 = setdiff([1,2],id2);
        end
    end
    X(n,:,1) = tmp(:,id1);
    X(n,:,2) = tmp(:,id2);
    cluster_allocation(n) = n_clust_id;
end
end

%% Figure with random data

function pointsspherefig(X,cluster_id)
gridPoints = 1000;
[XX,YY,ZZ] = sphere(gridPoints);
figure('units','normalized','outerposition',[0 0 .5 1]); clf;%'visible','off',

sh(1) = surf(XX,YY,ZZ, 'FaceAlpha', .2, 'EdgeAlpha', .1,'EdgeColor','none','FaceColor','none');
hold on; axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
view(-29,-13)

% smaller sphere to show lines on
[x2,y2,z2] = sphere(20); %30
sh(2) = surf(x2,y2,z2, 'EdgeAlpha', .2,'FaceColor','none','EdgeColor',[0,0,0]);
set(gca,'XColor', 'none','YColor','none','ZColor','none')
grid off
view(-29,-13)

cols = [0,0.4,0;0.5,0,0;0,0,0.5];

for i = 1:numel(unique(cluster_id))
    scatter3(X(cluster_id==i,1), X(cluster_id==i,2), X(cluster_id==i,3),7,cols(i,:),'filled');
end
end