clear
% load random 50.000 points
% X_test = h5read('/dtu-compute/HCP_dFC/data/eigs_RL1_pca1000_big2.h5',...
%     '/Dataset',[657601,1],[inf,1000]);%
% X_test = h5read('/dtu-compute/HCP_dFC/data/eigs_RL1.h5','/Dataset',[657601,1],[inf,inf]);
% V = h5read('/dtu-compute/HCP_dFC/data/eigs_RL1_pca1000_big_half_svdV.h5','/Dataset',[1,1],[inf,inf]);
% X_test = X_test*V;
% X_test = X_test./vecnorm(X_test,2,2);
% h5create('/dtu-compute/HCP_dFC/data/eigs_RL1_pca1000_big_half_svdXV_test.h5','/Dataset',size(X_test));
% h5write('/dtu-compute/HCP_dFC/data/eigs_RL1_pca1000_big_half_svdXV_test.h5','/Dataset',X_test,[1,1],size(X_test));

X_test = h5read('/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_atlas_RL2.h5','/Dataset',[1,1],[inf,inf]);
% X_test = X_test - X_test;

maxNumCompThreads('automatic');
% X_test = X_test(1:50000,:);
ll_train = nan(30,5);
ll_test = nan(30,5);

%addpath('/dtu-compute/HCP_dFC/toolboxes/cifti-matlab')
%template_cifti = cifti_read('/dtu-compute/HCP_dFC/data/alldata/100206_rfMRI_REST1_LR.nii');

for k = 2:30
    disp(['Working on k=',num2str(k)])
    dirk = dir(['/dtu-compute/HCP_dFC/2023/hcp_dfc/models/atlas/k',num2str(k),'_*.mat']);
    
    for repl = 1:numel(dirk)
        load(fullfile(dirk(repl).folder,dirk(repl).name));
        ll_train(k,repl) = results_interim.ll;
        
%         for kk = 1:k
%             prior(kk,1) = sum(results_interim.posterior(:,kk,repl))./size(X_test,1);
% %             prior(kk,1) = sum(results_interim.idx==kk)./size(X_test,1);
%             
%             b = template_cifti;
%             b.cdata = (results_interim.mu(:,kk,repl)'*V')';
%             b.diminfo{2} = cifti_diminfo_make_scalars(1);
%             cifti_write(b,['/dtu-compute/HCP_dFC/fulldatatest/centroids_dec/cen',num2str(k),'_',num2str(kk),'_repl',num2str(repl),'.dscalar.nii']);
%             
%         end
        
        addpath('/dtu-compute/HCP_dFC/2023/hcp_dfc/src/models')
        c = size(X_test,2)/2;
        M2 = kummer_log(1/2,c,results_interim.kappa',50000);
        Cp = gammaln(c)-log(2)-c*log(pi)-M2';
        logpdf = Cp + results_interim.kappa.*((results_interim.mu'*X_test').^2);
        
        % Then the density for every observation and component
        density = log(results_interim.pri) + logpdf;
        logsum_density = log(sum(exp(density-max(density))))+max(density);
        
        % then the log-likelihood for all observations and components
        ll_test(k,repl) = sum(logsum_density);
        
%         Beta = exp(density - logsum_density)';
%         %%%%% M-step, maximize the log-likelihood by updating variables
%         
%         Betasum = sum(Beta,1);
%         pri = (Betasum/size(X_test,1))';
%         
%         % Then the density for every observation and component
%         density = log(pri) + logpdf;
%         logsum_density = log(sum(exp(density-max(density))))+max(density);
%         
%         % then the log-likelihood for all observations and components
%         ll_test(k,repl) = sum(logsum_density);
        
        
    end
    
end

%save(['/dtu-compute/HCP_dFC/fulldatatest/lltesttrain_',date],'ll_train','ll_train')

lltrainmean = nanmean(ll_train,2);
lltestmean = nanmean(ll_test,2);
lltrainstd = nanstd(ll_train,[],2);
llteststd = nanstd(ll_test,[],2);

figure,
errorbar(1:30,lltrainmean,lltrainstd,'k-o','LineWidth',1.5),hold on
errorbar(1:30,lltestmean,llteststd,'k--o','LineWidth',1.5)
legend('train','test','Location','NorthWest')
ylabel('Log likelihood'),xlabel('Model order k')
% xlim([1.5, 10.5]),%ylim([1.975*10^10, 1.987*10^10])
set(gca,'FontSize',20)
print(gcf,['/dtu-compute/HCP_dFC/2023/hcp_dfc/docs/llfig_train_test_',date],'-dpng','-r300')
figure,
errorbar(1:30,lltrainmean,lltrainstd,'k-o','LineWidth',1.5)
legend('train')
print(gcf,['/dtu-compute/HCP_dFC/2023/hcp_dfc/docs/llfig_train__',date],'-dpng','-r300')
figure,
errorbar(1:30,lltestmean,llteststd,'k--o','LineWidth',1.5)
legend('test')
print(gcf,['/dtu-compute/HCP_dFC/2023/hcp_dfc/docs/llfig_test_',date],'-dpng','-r300')

%% make nifti


    
% V = niftiread('/dtu-compute/HCP_dFC/alldata/100206_rfMRI_REST1_LR.nii');
% info = niftiinfo('/dtu-compute/HCP_dFC/alldata/100206_rfMRI_REST1_LR.nii');

