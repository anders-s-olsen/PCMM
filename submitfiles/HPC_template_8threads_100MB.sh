#!/bin/sh
#BSUB -J HPCjob100MB
#BSUB -q hpc
#BSUB -R "rusage[mem=512MB]"
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
module load pandas
# h5py/3.10.0-python-3.10.13
# pip install -e .

python3 experiments/experiment_torchvsEM_synthetic.py MACG 0.1 dc
