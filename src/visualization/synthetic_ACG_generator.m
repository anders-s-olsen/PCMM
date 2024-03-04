clear,close all

%% ACG mixture
for K = [2,5,10]
    for p = [3,10,25]
        disp(['ACG p',num2str(p),' K',num2str(K)])
        if K>=p
            continue
        end
        
        SIGMAs = [];
        for k = 1:K/2+1
            sig = isotropic_covariance(p,k);
            SIGMAs = cat(3,SIGMAs,sig);
            if k>K/2
                continue
            end
            sig = anisotropic_covariance(p,p-k,p-k+1);
            SIGMAs = cat(3,SIGMAs,sig);
        end

        for k = 1:K
            writetable(array2table(SIGMAs(:,:,k)),['data/synthetic/centroids/synth_cov_p',num2str(p),'K',num2str(K),'_',num2str(k),'.csv'],'WriteVariableNames',false)
        end

        idx = repelem(1:K,10000/K);


        % train data
        [X,cluster_id] = syntheticACGMixture(idx,SIGMAs,10000,0);
        % pointsspherefig(X,cluster_id);
        writetable(array2table(X),['data/synthetic/synth_data_ACG_p',num2str(p),'K',num2str(K),'_1.csv'],'WriteVariableNames',false)
        
        % test data
        [X,cluster_id] = syntheticACGMixture(idx,SIGMAs,10000,0);
        % pointsspherefig(X,cluster_id);
        writetable(array2table(X),['data/synthetic/synth_data_ACG_p',num2str(p),'K',num2str(K),'_2.csv'],'WriteVariableNames',false)

        %%%% MACG
        [X,cluster_id] = syntheticMACGMixture(idx,SIGMAs,10000,2,0);
        % pointsspherefig(X(:,:,1),cluster_id);
        % pointsspherefig(X(:,:,2),cluster_id);
        X2 = zeros(size(X,1),p);
        X2(1:2:20000,:) = X(:,:,1);
        X2(2:2:20000,:) = X(:,:,2);
        writetable(array2table(X2),['data/synthetic/synth_data_MACG_p',num2str(p),'K',num2str(K),'_1.csv'],'WriteVariableNames',false)
        [X,cluster_id] = syntheticMACGMixture(idx,SIGMAs,10000,2,0);
        % pointsspherefig(X(:,:,1),cluster_id);
        % pointsspherefig(X(:,:,2),cluster_id);
        X2 = zeros(2*size(X,1),p);
        X2(1:2:20000,:) = X(:,:,1);
        X2(2:2:20000,:) = X(:,:,2);
        writetable(array2table(X2),['data/synthetic/synth_data_MACG_p',num2str(p),'K',num2str(K),'_2.csv'],'WriteVariableNames',false)

    end
end
return

%% covariance matrices
function sig = isotropic_covariance(p,idx) %idx: which axis to concentrate around
sig = diag(ones(p,1)*1e-2);
sig(idx,idx)=1;
sig = p*sig/trace(sig);
end
function sig = anisotropic_covariance(p,idx1,idx2) %idx: which two axes to covary
diagonal = ones(p,1)*1e-2;
diagonal([idx1,idx2]) = 1;
offdiagonal = zeros(p);
offdiagonal(idx1,idx2) = 0.9;
offdiagonal(idx2,idx1) = 0.9;
sig = diag(diagonal)+offdiagonal;
sig = p*sig/trace(sig);
end

%% ACG sampler
function [X,cluster_allocation] = syntheticACGMixture(idx,SIGMAs,num_points,noise)

num_clusters = size(SIGMAs,3);
point_dim = size(SIGMAs,2);

X = zeros(num_points,point_dim);
cluster_allocation = zeros(num_points,1);
for n = 1:num_points
    x_i = mvnrnd(zeros(point_dim,1),SIGMAs(:,:,idx(n)),1);
    X(n,:) = x_i/norm(x_i);
    % nx = chol(SIGMAs(:,:,n_clust_id),'lower') * randn(point_dim,1)+noise*randn(point_dim,1);
    % X(n,:) = nx/norm(nx);
    cluster_allocation(n) = idx(n);
end
end

%% MACG sampler
function [X,cluster_allocation] = syntheticMACGMixture(idx,SIGMAs,num_points,num_cols,noise)

num_clusters = size(SIGMAs,3);
point_dim = size(SIGMAs,2);

target = [0,1;1,1;1,1];

X = zeros(num_points,point_dim,num_cols);
cluster_allocation = zeros(num_points,1);
for n = 1:num_points
    Xi = mvnrnd(zeros(point_dim,1),SIGMAs(:,:,idx(n)),num_cols)';
    Xsq = (Xi'*Xi)^(-0.5);
    tmp = Xi*Xsq;
    % sort
    % sim = (tmp'*(target./vecnorm(target))).^2;
    % [val1,id1] = max(sim(1,:));
    % [val2,id2] = max(sim(2,:));
    % if id1==id2
    %     if val1>val2
    %         id2 = setdiff([1,2],id1);
    %     else
    %         id1 = setdiff([1,2],id2);
    %     end
    % end
    X(n,:,1) = tmp(:,1);
    X(n,:,2) = tmp(:,2);
    cluster_allocation(n) = idx(n);
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