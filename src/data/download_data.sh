#!/bin/bash

subjectlist=/dtu-compute/HCP_dFC/2023/hcp_dfc/data/subjectlist.txt

while read -r subject;
do
    # fMRI
    mkdir -p /dtu-compute/HCP_dFC/2023/hcp_dfc/data/raw/${subject:4:6}/fMRI
    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/${subject:4:6}/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii \
        /dtu-compute/HCP_dFC/2023/hcp_dfc/data/raw/${subject:4:6}/fMRI/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii
    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/${subject:4:6}/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii \
        /dtu-compute/HCP_dFC/2023/hcp_dfc/data/raw/${subject:4:6}/fMRI/rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii
    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/${subject:4:6}/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii \
        /dtu-compute/HCP_dFC/2023/hcp_dfc/data/raw/${subject:4:6}/fMRI/rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii
    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/${subject:4:6}/MNINonLinear/Results/rfMRI_REST2_RL/rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii \
        /dtu-compute/HCP_dFC/2023/hcp_dfc/data/raw/${subject:4:6}/fMRI/rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii

#     # MEG
#     mkdir -p /dtu-compute/HCP_dFC/2023/hcp_dfc/data/raw/${subject:4:6}/MEG/anatomy
#     mkdir -p /dtu-compute/HCP_dFC/2023/hcp_dfc/data/raw/${subject:4:6}/MEG/Restin/icablpenv
# 
#     aws s3 sync \
#         s3://hcp-openaccess/HCP_1200/${subject:4:6}/MEG/anatomy/ \
#         /dtu-compute/HCP_dFC/2023/hcp_dfc/data/raw/${subject:4:6}/anatomy
#     filename3="_MEG_3-Restin_icablpenv_whole.power.dtseries.nii"
#     filename4="_MEG_4-Restin_icablpenv_whole.power.dtseries.nii"
#     filename5="_MEG_5-Restin_icablpenv_whole.power.dtseries.nii"
#     aws s3 cp \
#         s3://hcp-openaccess/HCP_1200/${subject:4:6}/MEG/Restin/icablpenv/${subject:4:6}$filename3 \
#         /dtu-compute/HCP_dFC/2023/hcp_dfc/data/raw/${subject:4:6}/Restin/icablpenv/${subject:4:6}$filename3
#     aws s3 cp \
#         s3://hcp-openaccess/HCP_1200/${subject:4:6}/MEG/Restin/icablpenv/${subject:4:6}$filename4 \
#         /dtu-compute/HCP_dFC/2023/hcp_dfc/data/raw/${subject:4:6}/Restin/icablpenv/${subject:4:6}$filename4
#     aws s3 cp \
#         s3://hcp-openaccess/HCP_1200/${subject:4:6}/MEG/Restin/icablpenv/${subject:4:6}$filename5 \
#         /dtu-compute/HCP_dFC/2023/hcp_dfc/data/raw/${subject:4:6}/Restin/icablpenv/${subject:4:6}$filename5



done < $subjectlist

