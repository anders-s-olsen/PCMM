#!/bin/bash

# check that current directory has a subfolder called "paper/"
if [ ! -d "paper" ]; then
    echo "This script must be run from the root of the repository."
    exit 1
fi

subjectlist=paper/data/255unrelatedsubjectsIDs.txt

# while read -r subject;do
while IFS= read -r subject || [[ -n "$subject" ]]; do
    # rfMRI
    # mkdir -p paper/data/raw/$subject/fMRI
    # aws s3 cp \
    #     s3://hcp-openaccess/HCP_1200/${subject:4:6}/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii \
    #     paper/data/raw/${subject:4:6}/fMRI/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii
    # aws s3 cp \
    #     s3://hcp-openaccess/HCP_1200/${subject:4:6}/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii \
    #     paper/data/raw/${subject:4:6}/fMRI/rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii
    # aws s3 cp \
    #     s3://hcp-openaccess/HCP_1200/$subject/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii \
    #     paper/data/raw/$subject/fMRI/rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii
    # aws s3 cp \
    #     s3://hcp-openaccess/HCP_1200/$subject/MNINonLinear/Results/rfMRI_REST2_RL/rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii \
    #     paper/data/raw/$subject/fMRI/rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii

    for task in MOTOR SOCIAL EMOTION GAMBLING LANGUAGE RELATIONAL WM; do
        mkdir -p paper/data/raw/$subject/fMRI
        aws s3 cp \
            s3://hcp-openaccess/HCP_1200/$subject/MNINonLinear/Results/tfMRI_${task}_RL/tfMRI_${task}_RL_Atlas.dtseries.nii \
            paper/data/raw/$subject/fMRI/tfMRI_${task}_RL_Atlas.dtseries.nii
        mkdir -p paper/data/raw/$subject/EVs/${task}
        aws s3 cp \
            s3://hcp-openaccess/HCP_1200/$subject/MNINonLinear/Results/tfMRI_${task}_RL/EVs/ \
            paper/data/raw/$subject/EVs/${task}/ --recursive
    done    

done < $subjectlist

