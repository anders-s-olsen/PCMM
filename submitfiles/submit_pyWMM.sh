#!/bin/sh
#BSUB -J HCPjob
#BSUB -q computebigbigmem
#BSUB -R "rusage[mem=20GB]"
#BSUB -B
#BSUB -N
#BSUB -o HCPjob_out_%J.txt
#BSUB -e HCPjob_err_%J.txt
#BSUB -W 168:00 
#BSUB -n 16
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 

source /dtu-compute/macaroni/miniconda3/bin/activate
conda activate hcp

cd ..
cd experiments

python3 run_WMM.py