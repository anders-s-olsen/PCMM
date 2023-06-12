function results = WMM_EM_BigMem2(X,K,maxIter,nRepl,init,neg,savefolder)
%
% Watson Mixture Model as in Sra2012 algorithm1, expectation-maximization.
% Mandatory input: nxp data X and number of clusters K
% Optional input: maxIter (default 1000), nRepl (5), initilization algo,
% negative concentration parameter (1/0)
% initialization as either uniform, diametricalkmeans++ (default) or
% diametricalkmeans which is itself initialized with diametricalkmeans++.
% Final input is whether lambda should be negative (1/0 [0])
% Output: [idx_out,mu_out,lambda_out,ll_out].
% Distance measure is the squared sample-to-sample angle.

rng shuffle

X_range = [min(min(X)),max(max(X))];

[n,p] = size(X);
if nargin == 2
    maxIter = 100;
    nRepl = 5;
    init = '++';
    neg = 0;
    savefolder = 'models/';
elseif nargin == 3
    nRepl = 5;
    init = '++';
    neg = 0;
    savefolder = 'models/';
elseif nargin == 4
    init = '++';
    neg = 0;
    savefolder = 'models/';
elseif nargin == 5
    neg = 0;
    savefolder = 'models/';
elseif nargin == 6
    savefolder = 'models/';
end
if neg
    tolerance = (10e-8)*numel(X);
    x0 = -1;
else
    tolerance = (10e-8)*numel(X);
    x0 = 1;
end
a = 1/2;
c = p/2;
options = optimoptions('fmincon','Display','off',...
                'MaxIterations',50,'OptimalityTolerance',1e-06);

% preallocate
mu_final = zeros(p,K,nRepl);
ll_final = zeros(1,nRepl);
kappa_final = zeros(K,nRepl);
X_part_final = zeros(n,K,nRepl);
pri_final = zeros(K,nRepl);
ll_iter_final = cell(1,nRepl);
it_final = zeros(1,nRepl);

%% perform clustering

for repl = 1:nRepl
%     clearvars -except X_range n p maxIter nRepl init neg tolerance a c K ...
%         mu_final ll_final lambda_final X_part_final X repl...
%         x0 options
    
%     mu = zeros(p,K,2);
%     kappa = zeros(K,2);
%     Beta = zeros(K,n,2);
%     loglik = zeros(1,2);
    
    mu = zeros(p,K);
    kappa = zeros(K);
    Beta = zeros(n,K);
    loglik = zeros(1);
    mu_old = zeros(p,K);
    kappa_old = zeros(K);
    Beta_old = zeros(K,n);
    loglik_old = zeros(1);
    
    if strcmp(init,'uniform')
        
        % Initilize clusters
        mu = unifrnd(X_range(1),X_range(2),p,K);
        mu = mu./norm(mu);
        
    elseif strcmp(init,'++')
        mu = pdfc_diametrical_clustering_plusplus(X,K);
        
    elseif strcmp(init,'diam')
        
        [~,mu,~] = pdfc_diametrical_clustering(X,K,maxIter,1,'++');
        
    end
    
    
    % Initialize cluster concentrations to be one or -1
    if neg == 0
        kappa = repelem(1,K)';
    elseif neg == 1
        kappa = repelem(-1,K)';
    end
    
    % Initialize cluster prior probabilities but it's pi so pri for prior
    pri = repelem(1/K,K)';
    loglik_iter = nan(1,maxIter);
    
    it = 0;
    T_tic = tic;
    while true
        
        % reassign newly calculated variables
        it = it + 1;
        mu_old = mu;
        Beta_old = Beta;
        kappa_old = kappa;
        loglik_old = loglik;
        pri_old = pri;
        
        
        %%%%% E-step, compute expectations %%%%%
        % first the pdf for every observation and component
        M2 = kummer_log(a,c,kappa',50000);
        Cp = gammaln(c)-log(2)-c*log(pi)-M2';
        logpdf = Cp + kappa.*((mu'*X').^2);
        
        % Then the density for every observation and component
        density = log(pri_old) + logpdf;
        logsum_density = log(sum(exp(density-max(density))))+max(density);
        
        % then the log-likelihood for all observations and components
        loglik = sum(logsum_density);
        loglik_iter(it) = loglik;
        
        % End while-loop if tolerance met or max-iterations
        if it>1
            diffll = diff([loglik_old,loglik]);
            if abs(diffll)<tolerance || it==maxIter %||loglik<loglik_old
                
                % update mu and lambda and loglik to the best one
                [~,m] = max([loglik_old,loglik]);
                
                % Output final parameters and partition
                if m == 1
                    mu_final(:,:,repl) = mu_old;
                    ll_final(:,repl) = loglik_old;
                    kappa_final(:,repl) = kappa_old;
                    pri_final(:,repl) = pri_old;
                
                    X_part_final(:,:,repl) = Beta_old;
                    it_final(repl) = it-1;
                else
                    mu_final(:,:,repl) = mu;
                    ll_final(:,repl) = loglik;
                    kappa_final(:,repl) = kappa;
                    pri_final(:,repl) = pri;
                
                    X_part_final(:,:,repl) = Beta;
                    it_final(repl) = it;
                end
                
                ll_iter_final{repl} = loglik_iter(~isnan(loglik_iter));
                
                results_interim.ll = ll_final;
                results_interim.mu = mu_final;
                results_interim.kappa = kappa_final;
                results_interim.posterior = X_part_final;
                results_interim.pri = pri_final;
                results_interim.loglik = ll_iter_final;
                results_interim.it = it_final;
                results_interim.input_data_size = size(X);
                
                num_saves = dir([savefolder,'k',num2str(K),'_Repl*']);
                % save([savefolder,'k',num2str(K),'_Repl',num2str(numel(num_saves)+1),'_',date],'results_interim')
                break
            end
        end
        
        % Then the assignment (posterior) probability
        Beta = exp(density - logsum_density)';
        
        
        %%%%% M-step, maximize the log-likelihood by updating variables
        
        Betasum = sum(Beta,1);
        pri = (Betasum/n)';
        for k = 1:K
            
            % Update mean directions as the eigenvalue of weighted scatter
            % we use Q instead for improved speed in high dimensions
            
            Q = sqrt(Beta(:,k)).*X;
            
            if kappa>0
                [~,~,mu(:,k)]=svds(Q,1,'largest','RightStartVector',mu_old(:,k));
                %                 tic,svds(Q,1);toc
                %                 tic
%                 if ismember(it,[1:50:maxIter,maxIter])
%                     [~,~,mu(:,k)] = svds(Q,1,'largest','RightStartVector',mu_old(:,k));
%                 else
%                     mutmp = mu(:,k);
%                     for t = 1:10
%                         mutmp = Q'*(Q*mutmp);
%                         mutmp = mutmp/norm(mutmp);
%                     end
%                     mu(:,k) = mutmp;
%                 end
%                 disp(num2str(it))
                %                 [~,~,mu(:,k)]=rsvd(Q,1);
                %                 T3 = toc;
                %                 disp(['Old svd took ',num2str(T3),' seconds'])
            elseif kappa<0
                [~,~,eigenvectors]=svds(Q,p); %Option smallest
                mu(:,k) = eigenvectors(:,end);
            end
            
%             rk = 1/sum(Beta(k,:))*(mu(:,k)'*Q')*(mu(:,k)'*Q')';
            rk = 1/sum(Beta(:,k))*sum((mu(:,k)'*Q').^2);
            
            
            % Now we turn to updating the precision variable, kappa
            % To this end, we optimize the component-wise log-likelihood
            % (Sra2012 eq 2.4) rewritten for one component of a mixture
            % l = kappa*rk-ln(M(a,c,kappa))
            
            
%             (1/Betasum(k)*(mu(:,k,2).*X)'*X);
            
            % Bounds on the solution derived by Sra2012
            LB = (rk*c-a)/(rk*(1-rk))*(1+(1-rk)/(c-a));
            B  = (rk*c-a)/(2*rk*(1-rk))*(1+sqrt(1+4*(c+1)*rk*(1-rk)/(a*(c-a))));
            UB = (rk*c-a)/(rk*(1-rk))*(1+rk/a);
            
            % Optimize kappa within bounds (always convex within bounds)
%             f = @(kappa)-(kappa*rk-log(KummerSimple(a,c,kappa)));
            f = @(kappa)-(kappa*rk-kummer_log(a,c,kappa,50000));
            f2 = @(kappa)abs((a/c)*(kummer_log(a+1,c+1,kappa,50000)/kummer_log(a,c,kappa,50000))-rk);
            
            if rk<1&&rk>a/c
                kappa(k) = fmincon(f,mean([LB,B]),[],[],[],[],LB,B,[],options);
                kappa(k) = fmincon(f2,mean([LB,B]),[],[],[],[],LB,B,[],options);
            elseif rk<a/c&&rk>0
                kappa(k) = fmincon(f,mean([B,UB]),[],[],[],[],B,UB,[],options);
            elseif rk==a/c
                kappa(k) = 0;
            else
%                 keyboard
                error('Kappa cannot be calculated, try normalizing input data')
            end
            
            if kappa(k)<eps
                kappa = eps;
            end
            
        end
        T_toc = toc(T_tic);
%         save(['loglik_',num2str(K),'_it',num2str(it)],'loglik')
        disp(['Done with iteration ',num2str(it),' in ',num2str(T_toc/60),' minutes'])
        
        
    end
    %     fprintf('\n'); % To go to a new line after reaching 100% progress
    % check memory
%         d=whos;
%         [Z,~] = size(d);
%         memory_size = 0;
%         for p = 1:Z
%             memory_size = memory_size + d(p).bytes;
%             disp(num2str(d(p).bytes))
%         end
%         memory_size = 9.31e-10.*memory_size;
end


% Output result from best replicate
% if nRepl>1
%     [~,idx_ll] = max(ll_final,[],2);
%     results.ll = ll_final(:,idx_ll);
%     results.mu = mu_final(:,:,idx_ll)';
%     results.lambda = lambda_final(:,idx_ll)';
%     results.idx = X_part_final(:,idx_ll);
%     results.pri = pri_final(:,idx_ll);
%     results.loglik = ll_iter_final{idx_ll};
%     results.it = it_final(idx_ll);
% else
    results.ll = ll_final;
    results.mu = mu_final;
    results.kappa = kappa_final;
    results.posterior = X_part_final;
    results.pri = pri_final;
    results.loglik = ll_iter_final;
    results.it = it_final;
    results.input_data_size = size(X);
% end

end






