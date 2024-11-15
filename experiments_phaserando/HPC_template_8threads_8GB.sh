#!/bin/sh
#BSUB -J HCPjob8GB
#BSUB -q hpc
#BSUB -R "rusage[mem=2GB]"
# #BSUB -B
# #BSUB -N
#BSUB -o experiments_phaserando/output/HCPjob_out_%J.txt
#BSUB -e experiments_phaserando/output/HCPjob_err_%J.txt
#BSUB -W 48:00 
#BSUB -n 2
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
# display information
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC

source ~/miniconda3/bin/activate
conda activate hcp
python3 experiments_phaserando/experiment_all_OHBM.py Normal phase_narrowband_controlled Kmeansseg
