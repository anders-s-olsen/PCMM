#!/bin/sh
#BSUB -J HCPjob8GB
#BSUB -q hpc
#BSUB -R "rusage[mem=8GB]"
# #BSUB -B
# #BSUB -N
#BSUB -o experiments_phaserando/output/HCPjob_out_%J.txt
#BSUB -e experiments_phaserando/output/HCPjob_err_%J.txt
#BSUB -W 72:00 
#BSUB -n 8
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC

source /dtu-compute/macaroni/miniconda3/bin/activate
conda activate hcp
module load pandas
module load h5py/3.10.0-python-3.10.13

python3 experiments_phaserando/experiment_phase_controlled.py MACG 0.1
