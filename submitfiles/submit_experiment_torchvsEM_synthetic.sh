#!/bin/sh
inits=("unif" "++" "dc")
modelnames=("Watson" "ACG" "MACG")
# bsub < submitfiles/pipinstall.sh
# sleep 30
for LR in 0 0.1
do
for init in "${inits[@]}"
do
for m in "${modelnames[@]}"
do

    sed -i '$ d' submitfiles/HPC_template_8threads_100MB.sh
    echo "python3 experiments/experiment_torchvsEM_synthetic.py $m $LR $init" >> submitfiles/HPC_template_8threads_100MB.sh
    bsub < submitfiles/HPC_template_8threads_100MB.sh

done
done
done