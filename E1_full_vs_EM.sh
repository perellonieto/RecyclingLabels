#!/bin/bash

source ./venv3/bin/activate

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

for m in "${m_list[@]}"
do
    for b in "${m_b_list[@]}"
    do
        python compare_full_vs_EM.py "${m}" 1.0 "${b}"
    done
done
