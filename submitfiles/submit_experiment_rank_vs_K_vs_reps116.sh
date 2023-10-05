#!/bin/sh
modelnames=("ACG" "MACG")
inits=("unif" "++")
for init in "${inits[@]}"
do
for LR in 0
do
for m in "${modelnames[@]}"
do

    sed -i '$ d' submitfiles/HPC_template_8threads_100MB.sh
    echo "python3 experiments/experiment_rank_vs_K_vs_reps_116.py $m $LR $init" >> submitfiles/HPC_template_8threads_100MB.sh
    bsub < submitfiles/HPC_template_8threads_100MB.sh

done
done
done