#!/bin/sh
modelnames=("Watson" "ACG" "MACG")
inits=("unif" "++")
for LR in 0 0.1
do
for m in "${modelnames[@]}"
do
for init in "${inits[@]}"
do
for GSR in 0 1
do

    sed -i '$ d' submitfiles/pyspheremixtures454_template.sh
    echo "python3 experiments/run_models_116.py $m $LR $init $GSR" >> submitfiles/pyspheremixtures454_template.sh
    bsub < submitfiles/pyspheremixtures454_template.sh

done
done
done
done