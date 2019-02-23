#!/bin/bash

source ./venv/bin/activate

declare -a test_list=(
    0.7
    0.8
    0.9
    0.92
)

val_proportion='0.5'
max_epochs=2000

for random_seed in {1..100};
do
    for test_proportion in "${test_list[@]}"
    do
        echo "python full_vs_EM_any_dataset.py $random_seed webs $test_proportion $val_proportion"
        python full_vs_EM_any_dataset.py $random_seed webs \
            $test_proportion $val_proportion $max_epochs \
            > "webs_${random_seed}_${test_proportion}.out" \
            2> "webs_${random_seed}_${test_proportion}.err"
    done
done
