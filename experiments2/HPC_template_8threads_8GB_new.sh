#!/bin/sh
#BSUB -J HCPjob8GB
#BSUB -q hpc
#BSUB -R "rusage[mem=1GB]"
#BSUB -o experiments2/output/HCPjob_out_%J.txt
#BSUB -e experiments2/output/HCPjob_err_%J.txt
#BSUB -W 48:00 
#BSUB -n 8
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC

source ~/miniconda3/bin/activate
conda activate hcp

python3 experiments2/experiment_model_order_100.py Complex_ACG 10 25
