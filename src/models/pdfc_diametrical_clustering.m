function [idx_out,C_out,Obj_out] = pdfc_diametrical_clustering(X,K,maxIter,nRepl,init,seed)
%
% Diametrical clustering as in Sra2012 algorithm 2.
% Input:
%%% X - nxp data
%%% K - number of clusters K
% Optional input:
%%% maxIter (default 1000)
%%% nRepl (default 5)
%%% init (default '++', other option 'uniform')
% Output:
%%% idx - state sequence (closest centroid for all samples)
%%% C - computed centroids
%
% Sra2012: "The multivariate Watson distribution: Maximum-likelihood 
% estimation and other aspects", Sra S, Karp D, 2012.
%
% Anders S Olsen June - November 2021, Neurobiology Research Unit


[n,p] = size(X);
if nargin == 2
    maxIter = 1000;
    nRepl = 5;
    init = 'plus';
elseif nargin == 3
    nRepl = 5;
    init = 'plus';
elseif nargin == 4
    init = 'plus';
end

if ~isempty(seed)
    rng(seed)
    stream = RandStream.getGlobalStream();
    else stream = [];
end


%% Initialize variables 

C_final = zeros(p,K,nRepl);
objective_final = zeros(1,nRepl);
X_part_final = zeros(n,nRepl);

%% perform clustering

for repl = 1:nRepl
    clearvars objective partsum
    
    % Initilize clusters
    if strcmp(init,'uniform')
        X_range = [min(X(:)),max(X(:))];
        C = unifrnd(X_range(1),X_range(2),p,K);
    elseif strcmp(init,'plus')
        C = pdfc_diametrical_clustering_plusplus(X,K,stream);
    end
    
    it = 0;
    while true
        it = it + 1;
        % E-step, similarity between all samples and all centroids
        dis2 = (X*C).^2;
        [maxdis,X_part] = max(dis2,[],2);
        
        objective(it) = mean(maxdis);
        
        for k = 1:K
            partsum(it,k) = sum(X_part==k);
        end
        
        if it>1
            if isequal(partsum(it,:),partsum(it-1,:))|| (it == maxIter)
                C_final(:,:,repl) = C;
                objective_final(:,repl) = objective(end);
                X_part_final(:,repl) = X_part;
                break
            end
        end
        
        
        % M-step, update centroids according to allocated samples
        for k = 1:K
            idx_k = X_part==k;
            A2 = X(idx_k,:)'*X(idx_k,:);
            C(:,k) = A2*C(:,k)/norm(A2*C(:,k));
        end
        
        
    end
end


% Output result from best replicate
[~,idx_obj] = max(objective_final,[],2);
Obj_out = objective_final(:,idx_obj);
C_out = C_final(:,:,idx_obj)';
idx_out = X_part_final(:,idx_obj);

end
