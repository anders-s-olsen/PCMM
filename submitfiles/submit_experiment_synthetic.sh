#!/bin/sh
inits=("unif" "++" "dc")
modelnames=("Watson" "ACG" "MACG")
for LR in 0 0.001 0.01 0.1 1
do
for init in "${inits[@]}"
do
for m in "${modelnames[@]}"
do

    sed -i '$ d' submitfiles/HPC_template_8threads_100MB.sh
    echo "python3 experiments/experiment_synthetic.py $m $LR $init" >> submitfiles/HPC_template_8threads_100MB.sh
    bsub < submitfiles/HPC_template_8threads_100MB.sh

done
done
done