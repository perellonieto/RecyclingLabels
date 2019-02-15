#!/bin/bash

source ./venv/bin/activate

declare -a test_list=(
    0.6
    0.7
    0.8
    0.9
    0.95
    0.98
    0.99
)

dataset_name='webs'

for random_seed in {1..100};
do
    for test_proportion in "${test_list[@]}"
    do
        echo "python full_vs_EM_any_dataset.py $random_seed $dataset_name $test_proportion random_weak 1.0 0.5"
        python full_vs_EM_any_dataset.py $random_seed $dataset_name \
            $test_proportion random_weak 1.0 0.5 > "${dataset}_${random_seed}.out" \
                2> "${dataset}_${random_seed}.err"
    done
done
