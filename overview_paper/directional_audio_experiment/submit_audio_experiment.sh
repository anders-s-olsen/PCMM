#!/bin/bash
#BSUB -J PCMM_audio_experiment
#BSUB -q hpc
#BSUB -R "rusage[mem=16GB]"
#BSUB -o audio_experiment_%J.out
#BSUB -e audio_experiment_%J.err
#BSUB -W 48:00
#BSUB -n 1
#BSUB -R "span[hosts=1]"

set -euo pipefail

source ~/miniconda3/bin/activate
conda activate hcp
cd /dtu-compute/HCP_dFC/2023/hcp_dfc/overview_paper/directional_audio_experiment

# Reuse the already prepared paired scene when present. Model and consensus
# settings do not require another convolution/STFT pass.
if [[ ! -f output_paired/prepared.json ]]; then
    python -u experiment.py prepare --config config.json
fi

# The run itself atomically refreshes the two condition-specific performance
# PNGs after every completed (K, model, condition) fit, and refreshes each K=3
# learned-component car atlas as its families become available.
python -u experiment.py run --config config.json
