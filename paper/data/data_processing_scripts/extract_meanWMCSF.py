import nibabel as nib
import numpy as np


# load subjectlist
subjects = np.loadtxt('paper/data/255unrelatedsubjectsIDs.txt', dtype=str)

# loop through subjects
for subject in subjects:
    # loop through tasks
    for task in ['MOTOR', 'SOCIAL', 'EMOTION', 'GAMBLING', 'LANGUAGE', 'RELATIONAL', 'WM']:
        # define file paths
        fmri_file = f'paper/data/raw/{subject}/fMRI/tfMRI_{task}_RL.nii.gz'
        wm_mask_file = f'paper/data/raw/{subject}/ROIs/WMReg.2.nii.gz'
        csf_mask_file = f'paper/data/raw/{subject}/ROIs/CSFReg.2.nii.gz'

        # load fMRI data
        fmri_img = nib.load(fmri_file)
        fmri_data = fmri_img.get_fdata()

        ########## White matter ##########
        # load WM mask
        wm_mask_img = nib.load(wm_mask_file)
        wm_mask_data = wm_mask_img.get_fdata().astype(bool)

        # extract WM time series
        wm_time_series = fmri_data[wm_mask_data, :]

        # compute mean WM time series
        mean_wm_time_series = np.mean(wm_time_series, axis=0)

        # save mean WM time series to text file
        np.savetxt(f'paper/data/raw/{subject}/regressors/{task}_RL_WM.txt', mean_wm_time_series)


        ########## CSF ##########
        # load CSF mask
        csf_mask_img = nib.load(csf_mask_file)
        csf_mask_data = csf_mask_img.get_fdata().astype(bool)   

        # extract CSF time series
        csf_time_series = fmri_data[csf_mask_data, :]

        # compute mean CSF time series
        mean_csf_time_series = np.mean(csf_time_series, axis=0)

        # save mean CSF time series to text file
        np.savetxt(f'paper/data/raw/regressors/{task}_RL_CSF.txt', mean_csf_time_series)