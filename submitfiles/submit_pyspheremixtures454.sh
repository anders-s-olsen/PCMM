#!/bin/sh
for m in 0 1 2 3
do

    sed -i '$ d' submitfiles/pyspheremixtures454_template.sh
    echo "python3 experiments/run_models_454.py $m" >> submitfiles/pyspheremixtures454_template.sh
    bsub < submitfiles/pyspheremixtures454_template.sh

done