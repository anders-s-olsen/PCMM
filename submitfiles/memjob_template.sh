#!/bin/sh
#BSUB -J bigjob_100GB
#BSUB -q computebigbigmem
#BSUB -R "rusage[mem=100GB]"
# #BSUB -B
# #BSUB -N
#BSUB -o submitfiles/output_mem/memjob_out_%J.txt
#BSUB -e submitfiles/output_mem/memjob_err_%J.txt
#BSUB -W 12:00 
#BSUB -n 8
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 

source /dtu-compute/macaroni/miniconda3/bin/activate
conda deactivate
module load python3
module load numpy
module load scipy
module load h5py/3.10.0-python-3.10.13

python3 experiments/experiment_memory_100.py Watson 2
