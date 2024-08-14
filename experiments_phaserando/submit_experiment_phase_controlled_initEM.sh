#!/bin/sh
rm /dtu-compute/HCP_dFC/2023/hcp_dfc/experiments_phaserando/output/*.txt
# rm /dtu-compute/HCP_dFC/2023/hcp_dfc/data/results/torchvsEM_phase_controlled_results/*.csv
modelnames=("Watson" "ACG" "MACG" "SingularWishart")
# bsub < submitfiles/pipinstall.sh
# sleep 30
for m in "${modelnames[@]}"
do
    sed -i '$ d' experiments_phaserando/HPC_template_8threads_8GB.sh
    echo "python3 experiments_phaserando/experiment_phase_controlled_initALL_OHBM.py $m" >> experiments_phaserando/HPC_template_8threads_8GB.sh
    bsub < experiments_phaserando/HPC_template_8threads_8GB.sh

done

for m in "${modelnames[@]}"
do
    sed -i '$ d' experiments_phaserando/HPC_template_8threads_8GB.sh
    echo "python3 experiments_phaserando/experiment_phase_controlled_initKmeans_OHBM.py $m" >> experiments_phaserando/HPC_template_8threads_8GB.sh
    bsub < experiments_phaserando/HPC_template_8threads_8GB.sh

done

for m in "${modelnames[@]}"
do
    sed -i '$ d' experiments_phaserando/HPC_template_8threads_8GB.sh
    echo "python3 experiments_phaserando/experiment_phase_controlled_initKmeansseg_OHBM.py $m" >> experiments_phaserando/HPC_template_8threads_8GB.sh
    bsub < experiments_phaserando/HPC_template_8threads_8GB.sh

done

for m in "${modelnames[@]}"
do
    sed -i '$ d' experiments_phaserando/HPC_template_8threads_8GB.sh
    echo "python3 experiments_phaserando/experiment_phase_controlled_initrandom_OHBM.py $m" >> experiments_phaserando/HPC_template_8threads_8GB.sh
    bsub < experiments_phaserando/HPC_template_8threads_8GB.sh

done

# modelnames2=("euclidean" "diametrical" "grassmann" "weighted_grassmann")
# for m in "${modelnames2[@]}"
# do
#     sed -i '$ d' experiments_phaserando/HPC_template_8threads_8GB.sh
#     echo "python3 experiments_phaserando/experiment_phase_controlled_OHBM_kmeans.py $m" >> experiments_phaserando/HPC_template_8threads_8GB.sh
#     bsub < experiments_phaserando/HPC_template_8threads_8GB.sh

# done