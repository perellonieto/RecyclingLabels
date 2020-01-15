#!/bin/bash

source ./venv/bin/activate

declare -a train_prop_list=(
    0.1
    0.2
    0.4
    0.5
    0.6
    0.8
    0.9
    1.0
)

for random_state in {0..09};
do
    for prop in "${train_prop_list[@]}"
    do
        echo "
        python Example_19_full_em_osl_webs_incresing_true.py ${random_state} \
             ${prop} &> "Example_19_${random_state}_${prop}.out"
             "
        python Example_19_full_em_osl_webs_incresing_true.py ${random_state} \
             ${prop} &> "Example_19_${random_state}_${prop}.out"
    done
done
