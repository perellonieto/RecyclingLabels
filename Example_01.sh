#!/bin/bash

source ./venv/bin/activate

declare -a m_list=(
    noisy
    random_noise
    random_weak
    IPL
    quasi_IPL
)

declare -a m_b_list=(
    0.2
    0.3
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

dataset_name='diagonals'
output_path='results'`date +"_%Y_%m_%d"`

for random_state in {1..10};
do
    for prop in "${train_prop_list[@]}"
    do
        for m in "${m_list[@]}"
        do
            for b in "${m_b_list[@]}"
            do
                 echo "python Example_01_full_em_osl.py --m-method ${m} --beta ${b} \
                     --train-proportion ${prop} --random-state ${random_state} \
                     --out-folder ${output_path} --max-epochs 1000 --redirect-std"
                 python Example_01_full_em_osl.py --m-method ${m} --beta ${b} \
                     --train-proportion ${prop} --random-state ${random_state} \
                     --out-folder ${output_path} --max-epochs 1000 --redirect-std
            done
        done
    done
done



