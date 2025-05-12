#!/bin/bash

datasets=('yacht' 'boston' 'energy' 'concrete' 'wine' 'power' 'kin8nm' 'naval' 'protein')
# losses=("calipso" "scaled_batch_cal" "batch_int" "batch_qr")
losses=("calipso_pretrain")
# seeds=(0 1 2 3 4)
seeds="$1"
visible_gpus="$2"
IFS="," read -r -a seeds_array <<< "$seeds"
echo "visible_gpus: $visible_gpus"
# Now you can access the elements using array indexing:
echo "seeds: ${seeds_array[@]}"
for data in "${datasets[@]}"; do
    for loss in "${losses[@]}"; do
        for seed in "${seeds_array[@]}"; do
            echo "Running $loss on $data with seed $seed"
            python main.py --loss "$loss" --data "$data" --seed "$seed" --gpu "$visible_gpus"
        done
    done
done