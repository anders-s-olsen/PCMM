#!/bin/sh
# rm /dtu-compute/HCP_dFC/2023/hcp_dfc/experiments_phaserando/output/*.txt
# rm /dtu-compute/HCP_dFC/2023/hcp_dfc/data/results/torchvsEM_phase_controlled_results/phase_controlled*.csv
modelnames=("Watson" "Complex_Watson" "ACG" "Complex_ACG" "MACG" "SingularWishart")
dataset="phase_amplitude_controlled"
# bsub < submitfiles/pipinstall.sh
# sleep 30
for m in "${modelnames[@]}"
do
    sed -i '$ d' experiments_phaserando/HPC_template_8threads_8GB.sh
    echo "python3 experiments_phaserando/experiment_phase_controlled_initALL_OHBM.py $m $dataset" >> experiments_phaserando/HPC_template_8threads_8GB.sh
    bsub < experiments_phaserando/HPC_template_8threads_8GB.sh
done

for m in "${modelnames[@]}"
do
    sed -i '$ d' experiments_phaserando/HPC_template_8threads_8GB.sh
    echo "python3 experiments_phaserando/experiment_phase_controlled_initKmeans_OHBM.py $m $dataset" >> experiments_phaserando/HPC_template_8threads_8GB.sh
    bsub < experiments_phaserando/HPC_template_8threads_8GB.sh
done

for m in "${modelnames[@]}"
do
    sed -i '$ d' experiments_phaserando/HPC_template_8threads_8GB.sh
    echo "python3 experiments_phaserando/experiment_phase_controlled_initKmeansseg_OHBM.py $m $dataset" >> experiments_phaserando/HPC_template_8threads_8GB.sh
    bsub < experiments_phaserando/HPC_template_8threads_8GB.sh
done

for m in "${modelnames[@]}"
do
    sed -i '$ d' experiments_phaserando/HPC_template_8threads_8GB.sh
    echo "python3 experiments_phaserando/experiment_phase_controlled_initrandom_OHBM.py $m $dataset" >> experiments_phaserando/HPC_template_8threads_8GB.sh
    bsub < experiments_phaserando/HPC_template_8threads_8GB.sh
done