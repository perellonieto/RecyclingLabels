#!/bin/bash

source ./venv/bin/activate

declare -a m_list=(
    noisy
    random_noise
    random_weak
    IPL
    quasi_IPL
)

declare -a m_a_list=(
    0.0
    0.2
    0.5
    0.8
    1.0
)

dataset_name='blobs'
test_proportion='0.8'
val_proportion='0.5'

for random_seed in {1..10};
do
    for m in "${m_list[@]}"
    do
        for a in "${m_a_list[@]}"
        do
            echo "python full_vs_EM_any_dataset.py $random_seed $dataset_name $test_proportion $m $a"
            python full_vs_EM_any_dataset.py $random_seed $dataset_name \
                $test_proportion $val_proportion $m $a > "${dataset_name}_${random_seed}_${m}_${a}.out" \
                2> "${dataset_name}_${random_seed}_${m}_${a}.err"
        done
    done
done
