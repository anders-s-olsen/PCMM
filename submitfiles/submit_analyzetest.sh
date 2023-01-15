#!/bin/sh
#BSUB -J analyzekjob
#BSUB -q computebigbigmem
#BSUB -R "rusage[mem=60GB]"
#BSUB -B
#BSUB -N
#BSUB -o analyzekjob_out_%J.txt
#BSUB -e analyzekjob_err_%J.txt
#BSUB -W 10:00 
#BSUB -n 8
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
module load matlab/R2020a
matlab -nodisplay -batch analyze_k_results