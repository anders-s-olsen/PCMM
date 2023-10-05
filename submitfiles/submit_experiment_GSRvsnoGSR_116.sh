#!/bin/sh
modelnames=("Watson" "ACG" "MACG")
inits=("unif" "++")
for init in "${inits[@]}"
do
for GSR in 0 1
do
for LR in 0 0.1 0.01
do
for m in "${modelnames[@]}"
do

    sed -i '$ d' submitfiles/HPC_template_8threads_8GB.sh
    echo "python3 experiments/experiment_GSRvsnoGSR_116.py $m $LR $init $GSR" >> submitfiles/HPC_template_8threads_8GB.sh
    bsub < submitfiles/HPC_template_8threads_8GB.sh

done
done
done
done