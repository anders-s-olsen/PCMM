#!/bin/sh
inits=("unif" "++" "dc")
for LR in 0 0.001 0.01 0.1 1
do
for init in "${inits[@]}"
do
for m in 0 1
do

    sed -i '$ d' submitfiles/pyspheremixtures_synth_template.sh
    echo "python3 experiments/run_models_synth.py $m $LR $init" >> submitfiles/pyspheremixtures_synth_template.sh
    bsub < submitfiles/pyspheremixtures_synth_template.sh

done
done
done