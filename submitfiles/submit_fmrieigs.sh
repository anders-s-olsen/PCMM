#!/bin/sh
#BSUB -J eigsjob
#BSUB -q hpc
#BSUB -R "rusage[mem=8GB]"
#BSUB -B
#BSUB -N
#BSUB -o eigsjob_out_%J.txt
#BSUB -e eigsjob_err_%J.txt
#BSUB -W 72:00 
#BSUB -n 16
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
module load matlab/R2020a
matlab -nodisplay -sd /dtu-compute/HCP_dFC/2023/hcp_dfc/src/data/ -batch fMRI_compute_eigs