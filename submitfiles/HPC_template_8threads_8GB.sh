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

python3 experiments/experiment_rank_vs_K_vs_reps_116.py MACG 0 ++
