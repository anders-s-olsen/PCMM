#!/bin/sh
#BSUB -J HPCjob
#BSUB -q computebigbigmem
#BSUB -R "rusage[mem=10GB]"
#BSUB -B
#BSUB -N
#BSUB -o HPCjob_out_%J.txt
#BSUB -e HPCjob_err_%J.txt
#BSUB -W 168:00 
#BSUB -n 16
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
module load matlab/R2020a
matlab -nodisplay -sd /dtu-compute/HCP_dFC/2023/hcp_dfc/src/models/ -batch run_WMM_EM_BigMem