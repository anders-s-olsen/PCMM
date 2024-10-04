#!/bin/sh
rm /dtu-compute/HCP_dFC/2023/hcp_dfc/experiments2/output/*.txt
rm /dtu-compute/HCP_dFC/2023/hcp_dfc/data/results/116_results/*.csv
rm /dtu-compute/HCP_dFC/2023/hcp_dfc/data/results/116_results/posteriors/*.txt
rm /dtu-compute/HCP_dFC/2023/hcp_dfc/data/results/116_results/lls/*.txt
rm /dtu-compute/HCP_dFC/2023/hcp_dfc/data/results/116_results/params/*.npy
modelnames=("Watson" "Complex_Watson" "ACG" "Complex_ACG" "MACG" "SingularWishart")
# modelnames=("Complex_ACG" "MACG" "SingularWishart")
# modelnames=("Complex_Watson")
# modelnames=("Complex_ACG")
for K in 1
do
for m in "${modelnames[@]}"
do
    sed -i '$ d' experiments2/HPC_template_8threads_8GB_new.sh
    echo "python3 experiments2/experiment_sequential_init_100.py $m $K" >> experiments2/HPC_template_8threads_8GB_new.sh
    bsub < experiments2/HPC_template_8threads_8GB_new.sh

done
done
