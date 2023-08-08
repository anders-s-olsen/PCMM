#!/bin/sh
inits=("unif" "++" "dc")
for K in {2..30}
do
for LR in 0 1 0 0.1 0.01
do
for init in "${inits[@]}"
do

    sed -i '$ d' submitfiles/pyspheremixtures454_template.sh
    echo "python3 experiments/run_models_454.py Watson $LR $init $K" >> submitfiles/pyspheremixtures454_template.sh
    bsub < submitfiles/pyspheremixtures454_template.sh

done
done
done