#!/bin/bash

# check that current directory has a subfolder called "paper/"
if [ ! -d "paper" ]; then
    echo "This script must be run from the root of the repository."
    exit 1
fi

subjectlist=paper/data/255unrelatedsubjectsIDs.txt

# while read -r subject;do
while IFS= read -r subject || [[ -n "$subject" ]]; do
    # rfMRI data
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
    
    # rfMRI and tfMRI regressors
    mkdir -p paper/data/raw/$subject/regressors
    for task in MOTOR SOCIAL EMOTION GAMBLING LANGUAGE RELATIONAL WM; do
        aws s3 cp \
            s3://hcp-openaccess/HCP_1200/$subject/MNINonLinear/Results/tfMRI_${task}_RL/Movement_Regressors_dt.txt \
            paper/data/raw/$subject/regressors/${task}_RL_Movement_Regressors_dt.txt
    done
    for rest in REST1 REST2; do
        aws s3 cp \
            s3://hcp-openaccess/HCP_1200/$subject/MNINonLinear/Results/rfMRI_${rest}_RL/Movement_Regressors_dt.txt \
            paper/data/raw/$subject/regressors/${rest}_RL_Movement_Regressors_dt.txt
        aws s3 cp \
            s3://hcp-openaccess/HCP_1200/$subject/MNINonLinear/Results/rfMRI_${rest}_RL/rfMRI_${rest}_RL_WM.txt \
            paper/data/raw/$subject/regressors/${rest}_RL_WM.txt
        aws s3 cp \
            s3://hcp-openaccess/HCP_1200/$subject/MNINonLinear/Results/rfMRI_${rest}_RL/rfMRI_${rest}_RL_CSF.txt \
            paper/data/raw/$subject/regressors/${rest}_RL_CSF.txt
    done

    # ROIs for extracting tfMRI mean WM and mean CSF timeseries
    mkdir -p paper/data/raw/$subject/ROIs
    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/$subject/MNINonLinear/ROIs/WMReg.2.nii.gz \
        paper/data/raw/$subject/ROIs/WMReg.2.nii.gz
    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/$subject/MNINonLinear/ROIs/CSFReg.2.nii.gz \
        paper/data/raw/$subject/ROIs/CSFReg.2.nii.gz

    for task in MOTOR SOCIAL EMOTION GAMBLING LANGUAGE RELATIONAL WM; do
        # data to model
        mkdir -p paper/data/raw/$subject/fMRI
        rm -f paper/data/raw/$subject/fMRI/tfMRI_${task}_RL_Atlas.dtseries.nii
        aws s3 cp \
            s3://hcp-openaccess/HCP_1200/$subject/MNINonLinear/Results/tfMRI_${task}_RL/tfMRI_${task}_RL_Atlas_MSMAll.dtseries.nii \
            paper/data/raw/$subject/fMRI/tfMRI_${task}_RL_Atlas_MSMAll.dtseries.nii
        
        # data to extract regressors from
        aws s3 cp \
            s3://hcp-openaccess/HCP_1200/$subject/MNINonLinear/Results/tfMRI_${task}_RL/tfMRI_${task}_RL.nii.gz \
            paper/data/raw/$subject/fMRI/tfMRI_${task}_RL.nii.gz
        
        # EVs
        mkdir -p paper/data/raw/$subject/EVs/${task}
        aws s3 cp \
            s3://hcp-openaccess/HCP_1200/$subject/MNINonLinear/Results/tfMRI_${task}_RL/EVs/ \
            paper/data/raw/$subject/EVs/${task}/ --recursive
    done    

done < $subjectlist

