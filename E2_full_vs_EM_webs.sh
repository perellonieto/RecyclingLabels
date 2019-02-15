#!/bin/bash

source ./venv/bin/activate

declare -a r_list=(
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
)

dataset_name='blobs'
test_proportion='0.8'

for random_seed in "${r_list[@]}"
do
    python full_vs_EM_any_dataset.py $random_seed $dataset_name \
        $test_proportion random_weak 1.0 0.5
done
