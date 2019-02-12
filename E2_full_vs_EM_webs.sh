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

for random_seed in "${r_list[@]}"
do
    python full_vs_EM_any_dataset.py $random_seed webs random_weak 1.0 0.5
done
