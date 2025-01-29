#!/bin/bash

datasets=('boston' 'concrete' 'energy' 'kin8nm' 'power' 'wine' 'naval')
losses=("batch_int" "batch_qr" "scaled_batch_cal")
seeds=(0 1 2 3 4)
for data in "${datasets[@]}"; do
    for loss in "${losses[@]}"; do
        #for seed in "${seeds[@]}"; do
        python main.py --loss "$loss" --data "$data" --seed 0
        #done
    done
done