#!/bin/sh
#BSUB -J HCPjob8GB
#BSUB -q hpc
#BSUB -R "rusage[mem=32GB]"
#BSUB -o experiments2/output/HCPjob_out_%J.txt
#BSUB -e experiments2/output/HCPjob_err_%J.txt
#BSUB -W 48:00 
#BSUB -n 2
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC

source ~/miniconda3/bin/activate
conda activate hcp
module load pandas
module load h5py/3.10.0-python-3.10.13

python3 experiments2/experiment_sequential_init_100.py SingularWishart 10
