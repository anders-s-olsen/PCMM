subjects = dir('/dtu-compute/HCP_dFC/2023/hcp_dfc/data/raw');

for sub = 1:numel(subjects)
    disp(['Starting1 ',num2str(sub)])
    dses = dir([subjects(sub).folder,'/',subjects(sub).name,'/fMRI/rfMRI_REST*_LR*']);
    for ses = 1:numel(dses)
        delete([dses(ses).folder,'/',dses(ses).name]);
    end
    
    disp(['Done with subject ',num2str(sub)])
end

dses = dir('/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_full/*rfMRI_REST*_LR*');
for ses = 1:numel(dses)
    disp(['Starting2 ',num2str(ses)])
    delete([dses(ses).folder,'/',dses(ses).name]);
    disp(['Done with session ',num2str(ses)])
end

dses = dir('/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_SchaeferTian116/*rfMRI_REST*_LR*');
for ses = 1:numel(dses)
    disp(['Starting3 ',num2str(ses)])
    delete([dses(ses).folder,'/',dses(ses).name]);
    disp(['Done with session ',num2str(ses)])
end

dses = dir('/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_SchaeferTian116_GSR/*rfMRI_REST*_LR*');
for ses = 1:numel(dses)
    disp(['Starting4 ',num2str(ses)])
    delete([dses(ses).folder,'/',dses(ses).name]);
    disp(['Done with session ',num2str(ses)])
end

dses = dir('/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_SchaeferTian454/*rfMRI_REST*_LR*');
for ses = 1:numel(dses)
    disp(['Starting5 ',num2str(ses)])
    delete([dses(ses).folder,'/',dses(ses).name]);
    disp(['Done with session ',num2str(ses)])
end