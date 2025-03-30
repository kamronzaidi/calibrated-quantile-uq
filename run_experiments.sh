#!/bin/bash

datasets=('yacht' 'boston' 'energy' 'concrete' 'wine' 'power' 'kin8nm' 'naval')
losses=("batch_int" "batch_qr" "scaled_batch_cal")
seeds=(0 1 2 3 4)
for data in "${datasets[@]}"; do
    for loss in "${losses[@]}"; do
        for seed in "${seeds[@]}"; do
            python main.py --loss "$loss" --data "$data" --seed "$seed"
        done
    done
done