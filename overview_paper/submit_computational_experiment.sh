#!/bin/bash
#BSUB -J PCMM_computational_experiment
#BSUB -q hpc
#BSUB -R "rusage[mem=16GB]"
#BSUB -o overview_paper/computational_experiment_%J.out
#BSUB -e overview_paper/computational_experiment_%J.err
#BSUB -W 48:00
#BSUB -n 1
#BSUB -R "span[hosts=1]"

source ~/miniconda3/bin/activate
conda activate hcp

python overview_paper/run_computational_experiment.py --resume
