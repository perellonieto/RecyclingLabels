#!/bin/bash

source ./venv/bin/activate

declare -a m_a_list=(
    0.9
    0.8
    0.7
    0.6
)

declare -a train_prop_list=(
    0.0
    0.1
    0.2
    0.3
    0.5
    0.7
    1.0
)

for random_state in {10..19};
do
    for prop in "${train_prop_list[@]}"
    do
        for a in "${m_a_list[@]}"
        do
            echo "
            python Example_09_full_em_osl_cifar10_multiple.py ${random_state} \
                 ${prop} ${a} 2>&1 > "Example_09_${random_state}_${prop}_${a}.out"
                 "
            python Example_09_full_em_osl_cifar10_multiple.py ${random_state} \
                 ${prop} ${a} &> "Example_09_${random_state}_${prop}_${a}.out"
        done
    done
done
