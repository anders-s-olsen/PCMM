#!/bin/sh
modelnames=("Watson" "ACG" "MACG")
for K in {2..30}
do
for LR in 0.1
do
for m in "${modelnames[@]}"
do
    sed -i '$ d' submitfiles/pyspheremixturesfull_template.sh
    echo "python3 experiments/run_models_full.py $m $LR ++ $K" >> submitfiles/pyspheremixturesfull_template.sh
    bsub < submitfiles/pyspheremixturesfull_template.sh
done
done
done