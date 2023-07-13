#!/bin/sh
#BSUB -J HCPjob
#BSUB -q hpc
#BSUB -R "rusage[mem=4GB]"
# #BSUB -B
# #BSUB -N
#BSUB -o submitfiles/output/HCPjob_out_%J.txt
#BSUB -e submitfiles/output/HCPjob_err_%J.txt
#BSUB -W 24:00 
#BSUB -n 8
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 

source /dtu-compute/macaroni/miniconda3/bin/activate
conda activate hcp

python3 experiments/run_models_454.py 0 1 dc
