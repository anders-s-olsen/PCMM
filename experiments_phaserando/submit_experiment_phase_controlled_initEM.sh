#!/bin/sh
rm /dtu-compute/HCP_dFC/2023/hcp_dfc/experiments_phaserando/output/*.txt
# rm /dtu-compute/HCP_dFC/2023/hcp_dfc/data/results/torchvsEM_phase_controlled_results/phase_controlled*.csv
modelnames=("Watson" "Complex_Watson" "ACG" "Complex_ACG" "MACG" "SingularWishart" "Normal")
dataset="phase_narrowband_controlled"
experiments=("random" "Kmeans" "ALL" "Kmeansseg") # 
# bsub < submitfiles/pipinstall.sh
# sleep 30
for experiment in "${experiments[@]}"
do
for m in "${modelnames[@]}"
do
    sed -i '$ d' experiments_phaserando/HPC_template_8threads_8GB.sh
    echo "python3 experiments_phaserando/experiment_all_OHBM.py $m $dataset $experiment" >> experiments_phaserando/HPC_template_8threads_8GB.sh
    bsub < experiments_phaserando/HPC_template_8threads_8GB.sh
done
done