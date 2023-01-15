#!/bin/sh
#BSUB -J MEGprocjob
#BSUB -q computebigbigmem
#BSUB -R "rusage[mem=64GB]"
#BSUB -B
#BSUB -N
#BSUB -o MEGprocjob_out_%J.txt
#BSUB -e MEGprocjob_err_%J.txt
#BSUB -W 168:00 
#BSUB -n 8
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
module load matlab/R2020a
matlab -nodisplay -sd /dtu-compute/HCP_dFC/2023/hcp_dfc/src/data/ -batch MEG_compute_phase