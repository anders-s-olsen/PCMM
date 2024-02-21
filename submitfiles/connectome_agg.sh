#!/bin/sh
#BSUB -J aggjob_31GB
#BSUB -q computebigbigmem
#BSUB -R "rusage[mem=31GB]"
# #BSUB -B
# #BSUB -N
#BSUB -o submitfiles/aggjob_out_%J.txt
#BSUB -e submitfiles/aggjob_err_%J.txt
#BSUB -W 72:00 
#BSUB -n 16
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 

source /dtu-compute/macaroni/miniconda3/bin/activate
conda activate hcp

python3 /dtu-compute/HCP_dFC/2023/hcp_dfc/connectome_agg/connectome_aggreg.py