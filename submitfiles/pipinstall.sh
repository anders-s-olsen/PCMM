#!/bin/sh
#BSUB -J pipinstalljob
#BSUB -q hpc
#BSUB -R "rusage[mem=128MB]"
# #BSUB -B
# #BSUB -N
#BSUB -o submitfiles/output/pipinstalljob_out_%J.txt
#BSUB -e submitfiles/output/pipinstalljob_err_%J.txt
#BSUB -W 1:00 
#BSUB -n 1
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 

source /dtu-compute/macaroni/miniconda3/bin/activate
conda activate hcp
pip install -e .