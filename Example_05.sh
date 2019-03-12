#!/bin/bash

source ./venv/bin/activate

declare -a m_b_list=(
    0.0
    0.2
    0.4
)

declare -a train_prop_list=(
    0.1
    0.2
    0.3
    0.5
    0.7
    1.0
)

dataset_name='supersetdigits'
output_path='results'`date +"_%Y_%m_%d"`
#output_path='results_2019_03_05'

for random_state in {1..50};
do
    for prop in "${train_prop_list[@]}"
    do
        for b in "${m_b_list[@]}"
        do
             echo "python Example_05_full_em_osl_supersetdigits.py ${random_state} ${prop} ${b} ${output_path}"
             python Example_05_full_em_osl_supersetdigits.py ${random_state} ${prop} ${b} ${output_path}
        done
    done
done



