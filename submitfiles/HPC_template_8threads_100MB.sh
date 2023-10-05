#!/bin/sh
#BSUB -J HPCjob100MB
#BSUB -q hpc
#BSUB -R "rusage[mem=100MB]"
# #BSUB -B
# #BSUB -N
#BSUB -o submitfiles/output/synthjob_out_%J.txt
#BSUB -e submitfiles/output/synthjob_err_%J.txt
#BSUB -W 24:00 
#BSUB -n 8
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 

source /dtu-compute/macaroni/miniconda3/bin/activate
conda activate hcp

python3 experiments/experiment_torchvsEM_synthetic.py MACG 1 dc
