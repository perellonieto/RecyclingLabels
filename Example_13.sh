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

weak_true_prop=0.07

for random_state in {0..09};
do
    for prop in "${train_prop_list[@]}"
    do
        for a in "${m_a_list[@]}"
        do
            echo "
            python Example_13_full_em_osl_mnist_complementary.py ${random_state} \
                 ${prop} ${weak_true_prop} 2>&1 > "Example_08_${random_state}_${prop}_${a}.out"
                 "
            python Example_13_full_em_osl_mnist_complementary.py ${random_state} \
                 ${prop} ${weak_true_prop} &> "Example_13_${random_state}_${prop}_${a}.out"
        done
    done
done
