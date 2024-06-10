#!/bin/sh
modelnames=("Watson" "ACG" "MACG")
# bsub < submitfiles/pipinstall.sh
# sleep 30
for m in "${modelnames[@]}"
do
    sed -i '$ d' experiments_phaserando/HPC_template_8threads_8GB.sh
    echo "python3 experiments_phaserando/experiment_phase_controlled_initEM.py $m" >> experiments_phaserando/HPC_template_8threads_8GB.sh
    bsub < experiments_phaserando/HPC_template_8threads_8GB.sh

done