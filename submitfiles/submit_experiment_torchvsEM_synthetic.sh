#!/bin/sh
inits=("unif" "++" "dc")
modelnames=("Watson" "ACG" "MACG")
for LR in 0 0.001 0.01 0.1 1
do
for init in "${inits[@]}"
do
for m in "${modelnames[@]}"
do

    sed -i '$ d' submitfiles/HPC_template_8threads_8GB.sh
    echo "python3 experiments/experiment_torchvsEM_454Watson.py $m $LR $init" >> submitfiles/HPC_template_8threads_8GB.sh
    bsub < submitfiles/HPC_template_8threads_8GB.sh

done
done
done