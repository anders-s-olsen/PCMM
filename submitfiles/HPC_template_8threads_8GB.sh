#!/bin/sh
#BSUB -J HCPjob8GB
#BSUB -q hpc
#BSUB -R "rusage[mem=8GB]"
# #BSUB -B
# #BSUB -N
#BSUB -o submitfiles/output/HCPjob_out_%J.txt
#BSUB -e submitfiles/output/HCPjob_err_%J.txt
#BSUB -W 72:00 
#BSUB -n 8
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 

source /dtu-compute/macaroni/miniconda3/bin/activate
conda activate hcp

python3 experiments/run_models_116.py MACG 0.01 ++ 1