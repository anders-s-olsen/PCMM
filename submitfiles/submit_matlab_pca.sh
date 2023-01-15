#!/bin/sh
#BSUB -J SVDjob
#BSUB -q computebigbigmem
#BSUB -R "rusage[mem=62GB]"
#BSUB -B
#BSUB -N
#BSUB -o SVDjob_out_%J.txt
#BSUB -e SVDjob_err_%J.txt
#BSUB -W 24:00 
#BSUB -n 16
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
module load matlab/R2020a
matlab -nodisplay -batch reduce_dimensionality