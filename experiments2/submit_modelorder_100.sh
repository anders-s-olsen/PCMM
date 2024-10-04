#!/bin/sh
# rm /dtu-compute/HCP_dFC/2023/hcp_dfc/experiments2/output/*.txt
# rm /dtu-compute/HCP_dFC/2023/hcp_dfc/data/results/116_results/*.csv
# rm /dtu-compute/HCP_dFC/2023/hcp_dfc/data/results/116_results/posteriors/*.txt
# rm /dtu-compute/HCP_dFC/2023/hcp_dfc/data/results/116_results/params/*.npy
# modelnames=("Complex_Watson")
# modelnames=("Complex_Watson" "Complex_ACG" "MACG" "SingularWishart")
modelnames=("Complex_ACG")
rank=25
for K in 1 2 3 4 5 6 7 8 9 10
do
for m in "${modelnames[@]}"
do
    sed -i '$ d' experiments2/HPC_template_8threads_8GB_new.sh
    echo "python3 experiments2/experiment_model_order_100.py $m $K $rank" >> experiments2/HPC_template_8threads_8GB_new.sh
    bsub < experiments2/HPC_template_8threads_8GB_new.sh

done
done
