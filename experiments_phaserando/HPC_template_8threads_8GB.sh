#!/bin/sh
#BSUB -J HCPjob8GB
#BSUB -q hpc
#BSUB -R "rusage[mem=1GB]"
# #BSUB -B
# #BSUB -N
#BSUB -o experiments_phaserando/output/HCPjob_out_%J.txt
#BSUB -e experiments_phaserando/output/HCPjob_err_%J.txt
#BSUB -W 48:00 
#BSUB -n 2
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
# display information
echo "exporting LSF_DJOB_NUMPROC"
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC

echo "sourcing miniconda"
source ~/miniconda3/bin/activate
echo "activating environment"
conda activate hcp
# echo "loading pandas"
# module load pandas
# echo "loading h5py"
# module load h5py

# cd /dtu-compute/HCP_dFC/2023/hcp_dfc
# conda env list
echo "running python script"
python3 experiments_phaserando/experiment_phase_controlled_initrandom_OHBM.py SingularWishart phase_narrowband_controlled
