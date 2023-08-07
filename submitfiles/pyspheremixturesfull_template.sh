#!/bin/sh
#BSUB -J bigjob
#BSUB -q computebigbigmem
#BSUB -R "rusage[mem=31GB]"
# #BSUB -B
# #BSUB -N
#BSUB -o submitfiles/output/bigjob_out_%J.txt
#BSUB -e submitfiles/output/bigjob_err_%J.txt
#BSUB -W 72:00 
#BSUB -n 16
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 

source /dtu-compute/macaroni/miniconda3/bin/activate
conda activate hcp

python3 experiments/run_models_454.py 2 0.1 ++ 30
