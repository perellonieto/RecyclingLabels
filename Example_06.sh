#!/bin/bash

source ./venv/bin/activate

declare -a m_a_list=(
    0.9
    0.7
    0.5
    0.3
    0.1
)

declare -a train_prop_list=(
    0.1
    0.2
    0.3
    0.5
    0.7
    1.0
)

dataset_name='digits'

for random_state in {1..50};
do
    for prop in "${train_prop_list[@]}"
    do
        for a in "${m_a_list[@]}"
        do
            echo "
                 python Example_06_full_em_osl_digits_multiple.py ${random_state}
                     ${prop} ${a}
                 "
            python Example_06_full_em_osl_digits_multiple.py ${random_state} \
                 ${prop} ${a} 2>&1 > "Example_06_${random_state}_${prop}_${a}.out"
        done
    done
done



