clear
% src = '/dtu-compute/HCP_dFC/2023/hcp_dfc/';
addpath(genpath(pwd))
% maxNumCompThreads('automatic');
data_train = h5read(['data/processed/fMRI_atlas_RL2.h5'],'/Dataset',[1,1],[240000,inf]);
data_test = h5read(['data/processed/fMRI_atlas_RL1.h5'],'/Dataset',[1,1],[240000,inf]);

[n,p] = size(data_train);
c = p/2;
name = 'Watson';

maxIter = 100000;
nRepl = 1;
neg = 0;
for ini = {'unif','++','dc'}
    if strcmp(ini,'unif')
        init = 'uniform';
    elseif strcmp(ini,'++')
        init = '++';
    elseif strcmp(ini,'dc')
        init = 'diam';
    end
    for K = [2,5,10]
        expname = ['454_',ini{1},'_-0.0_p',num2str(p),'_K',num2str(K)];
        for rep = 0:9 
            results = WMM_EM_BigMem2(data_train,K,maxIter,nRepl,init,neg);
            M2 = kummer_log(0.5,c,results.kappa',1000000);
            Cp = gammaln(c)-log(2)-c*log(pi)-M2';
            logpdf = Cp + results.kappa.*((results.mu'*data_test').^2);

            % Then the density for every observation and component
            density = log(results.pri) + logpdf;
            logsum_density = log(sum(exp(density-max(density))))+max(density);

            % then the log-likelihood for all observations and components
            loglik_test = sum(logsum_density);

            writetable(array2table([results.loglik{1}(end),loglik_test]),['experiments/454_outputs/',name,'_',expname,'_traintestlikelihood_r',num2str(rep),'.csv'])
        end
    end
end