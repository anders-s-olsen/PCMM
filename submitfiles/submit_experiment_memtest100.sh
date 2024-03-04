#!/bin/sh
modelnames=("Watson" "ACG" "MACG")
# bsub < submitfiles/pipinstall.sh
# sleep 30
for K in 2 5 10 15 20
do
for m in "${modelnames[@]}"
do

    sed -i '$ d' submitfiles/HPC_template_8threads_100MB.sh
    echo "python3 experiments/experiment_memory_100.py $m $K" >> submitfiles/HPC_template_8threads_100MB.sh
    bsub < submitfiles/HPC_template_8threads_100MB.sh

done
done