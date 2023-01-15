#!/bin/sh
#BSUB -J HPCjob
#BSUB -q computebigbigmem
#BSUB -R "rusage[mem=10GB]"
#BSUB -B
#BSUB -N
#BSUB -o threadsjob_out_%J.txt
#BSUB -e threadsjob_err_%J.txt
#BSUB -W 168:00 
#BSUB -n 37
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
module load matlab/R2020a
matlab -nodisplay -batch testthreads