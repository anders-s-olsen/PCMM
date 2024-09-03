#!/bin/sh
#BSUB -J HCPjob8GB
#BSUB -q hpc
#BSUB -R "rusage[mem=1GB]"
#BSUB -o experiments2/output/HCPjob_out_%J.txt
#BSUB -e experiments2/output/HCPjob_err_%J.txt
#BSUB -W 48:00 
#BSUB -n 1
#BSUB -R "span[hosts=1]"

source ~/miniconda3/bin/activate
conda activate hcp

cd /dtu-compute/HCP_dFC/2023/hcp_dfc
pip install -e .
