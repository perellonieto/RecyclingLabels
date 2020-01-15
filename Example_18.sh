#!/bin/bash

source ./venv/bin/activate

declare -a train_prop_list=(
    0.0
    0.1
    0.2
    0.3
    0.5
    0.7
    1.0
)

for random_state in {0..09};
do
    for prop in "${train_prop_list[@]}"
    do
        echo "
        python Example_18_full_em_osl_cifar10_complementary.py ${random_state} \
             ${prop} &> "Example_18_${random_state}_${prop}_${a}.out"
             "
        python Example_18_full_em_osl_cifar10_complementary.py ${random_state} \
             ${prop} &> "Example_18_${random_state}_${prop}_${a}.out"
    done
done
