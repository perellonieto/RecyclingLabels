#!/bin/bash

source ./venv/bin/activate

declare -a m_list=(
#    noisy
    random_noise
    random_weak
#    IPL
#    quasi_IPL
)

declare -a m_a_list=(
#    0.9
#    0.8
    0.7
#    0.6
#    0.5
)

declare -a dataset_list=(
    mnist
#    make_classification
#    digits
#    non_separable
#    separable
)

test_proportion='0.9'
val_proportion='0.3'
max_epochs=200

for random_seed in {1..50};
do
    for dataset_name in "${dataset_list[@]}"
    do
        for m in "${m_list[@]}"
        do
            for a in "${m_a_list[@]}"
            do
                echo "python full_vs_EM_any_dataset.py $random_seed $dataset_name $test_proportion $m $a"
                python full_vs_EM_any_dataset.py $random_seed $dataset_name \
                    $test_proportion $val_proportion $max_epochs $m $a > "${dataset_name}_${random_seed}_${m}_${a}.out" \
                    2> "${dataset_name}_${random_seed}_${m}_${a}.err"
            done
        done
    done
done
