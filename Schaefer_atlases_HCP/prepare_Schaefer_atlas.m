% This was used for Nina's project. The HCP project used Schaefer400
clear
atlases = {'100','200','300'};
addpath(genpath('/dtu-compute/HCP_dFC/2023/hcp_dfc/'))

load('/dtu-compute/HCP_dFC/2023/hcp_dfc/Schaefer_atlases_HCP/fs_LR_32k_medial_mask.mat')

for j = 1:numel(atlases)
    a=squeeze(niftiread(['/dtu-compute/HCP_dFC/2023/hcp_dfc/Schaefer_atlases_HCP/CBIG/',...
    'stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/',...
        'Schaefer2018_',atlases{j},'Parcels_7Networks_order.dlabel.nii']));
    a = a(medial_mask);
    numrois = numel(unique(a(a>0)));
    atlasmap = zeros(numrois,numel(a));
    for roi = 1:numrois
        atlasmap(roi,:) = a==roi;
    end
    save(['/dtu-compute/HCP_dFC/2023/hcp_dfc/Schaefer_atlases_HCP/Schaefer2018_',atlases{j},'Parcels_7Networks_map'],'atlasmap')
end


