#!/bin/bash

declare -a architectures=(
    'lr'
    'mlp100'
    'mlp100d'
    'mlp100d100d'
    )

declare -a models=(
    'EM'
    'OSL'
    'partially_weak'
    'fully_weak'
    )


for architecture in "${architectures[@]}"
do
    for model in "${models[@]}"
    do
        python ./run_baseline.py -d blobs -m "${model}" -a "${architecture}" \
            -r results_toy_example -s 0 -i 2 -k 2 -e -o -l mse
    done
done
