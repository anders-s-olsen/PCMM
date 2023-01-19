clear
src = '/dtu-compute/HCP_dFC/2023/hcp_dfc/';
maxNumCompThreads('automatic');
eigenvectors = h5read([src,'data/processed/fMRI_atlas_RL1.h5'],'/Dataset',[1,1],[inf,inf]);
savefolder = [src,'models/atlas/'];

maxIter = 500;
nRepl = 1;
init = '++';
neg = 0;
for k = 2:30
        %profile on
    results = WMM_EM_BigMem2(eigenvectors,k,maxIter,nRepl,init,neg,savefolder);
        %profile off
        %profsave
end