#!/bin/sh
inits=("unif" "++" "dc")
for LR in 0 1 0 0.1 0.01
do
for init in "${inits[@]}"
do

    sed -i '$ d' submitfiles/HPC_template_8threads_8GB.sh
    echo "python3 experiments/experiment_torchvsEM_454Watson.py Watson $LR $init" >> submitfiles/HPC_template_8threads_8GB.sh
    bsub < submitfiles/HPC_template_8threads_8GB.sh

done
done