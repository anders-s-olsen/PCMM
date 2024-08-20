#!/bin/sh
rm /dtu-compute/HCP_dFC/2023/hcp_dfc/experiments2/output/*.txt
# rm /dtu-compute/HCP_dFC/2023/hcp_dfc/data/results/torchvsEM_phase_controlled_results/*.csv
modelnames=("Watson" "ACG" "MACG" "SingularWishart")
# modelnames=("SingularWishart")
# bsub < submitfiles/pipinstall.sh
# sleep 30
for K in 1 4 10
do
for m in "${modelnames[@]}"
do
    sed -i '$ d' experiments2/HPC_template_8threads_8GB_new.sh
    echo "python3 experiments2/experiment_sequential_init_100.py $m $K" >> experiments2/HPC_template_8threads_8GB_new.sh
    bsub < experiments2/HPC_template_8threads_8GB_new.sh

done
done
# for m in "${modelnames[@]}"
# do
#     sed -i '$ d' experiments2/HPC_template_8threads_8GB.sh
#     echo "python3 experiments2/experiment_sequential_init_100.py $m" >> experiments2/HPC_template_8threads_8GB.sh
#     bsub < experiments2/HPC_template_8threads_8GB.sh

# done

# for m in "${modelnames[@]}"
# do
#     sed -i '$ d' experiments2/HPC_template_8threads_8GB.sh
#     echo "python3 experiments2/experiment_sequential_init_100.py $m" >> experiments2/HPC_template_8threads_8GB.sh
#     bsub < experiments2/HPC_template_8threads_8GB.sh

# done

# for m in "${modelnames[@]}"
# do
#     sed -i '$ d' experiments2/HPC_template_8threads_8GB.sh
#     echo "python3 experiments2/experiment_sequential_init_100.py $m" >> experiments2/HPC_template_8threads_8GB.sh
#     bsub < experiments2/HPC_template_8threads_8GB.sh

# done

# modelnames2=("euclidean" "diametrical" "grassmann" "weighted_grassmann")
# for m in "${modelnames2[@]}"
# do
#     sed -i '$ d' experiments2/HPC_template_8threads_8GB.sh
#     echo "python3 experiments2/experiment_phase_controlled_OHBM_kmeans.py $m" >> experiments2/HPC_template_8threads_8GB.sh
#     bsub < experiments2/HPC_template_8threads_8GB.sh

# done