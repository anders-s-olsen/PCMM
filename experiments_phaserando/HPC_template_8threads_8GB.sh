#!/bin/sh
#BSUB -J HCPjob8GB
#BSUB -q compute
#BSUB -R "rusage[mem=1GB]"
# #BSUB -B
# #BSUB -N
#BSUB -o experiments_phaserando/output/HCPjob_out_%J.txt
#BSUB -e experiments_phaserando/output/HCPjob_err_%J.txt
#BSUB -W 48:00 
#BSUB -n 8
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC

source ~/miniconda3/bin/activate
conda activate hcp
module load pandas
module load h5py/3.10.0-python-3.10.13

python3 experiments_phaserando/experiment_phase_controlled_initKmeans_OHBM.py ACG
