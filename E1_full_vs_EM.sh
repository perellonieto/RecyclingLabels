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

declare -a m_list=(
    noisy
    random_noise
    random_weak
    IPL
    quasi_IPL
)

declare -a m_b_list=(
    0.0
    0.2
    0.5
    0.8
    1.0
)

for random_seed in "${r_list[@]}"
do
    for m in "${m_list[@]}"
    do
        for b in "${m_b_list[@]}"
        do
            python full_vs_EM_any_dataset.py $random_seed blobs "${m}" 1.0 "${b}"
        done
    done
done
