#!/bin/sh
inits=("unif" "++" "dc")
for LR in 0.01 0.1 1
do
for init in "${inits[@]}"
do
for m in 0 1 2 3
do

    sed -i '$ d' submitfiles/pyspheremixtures454_template.sh
    echo "python3 experiments/run_models_454.py $m $LR $init" >> submitfiles/pyspheremixtures454_template.sh
    bsub < submitfiles/pyspheremixtures454_template.sh

done
done
done