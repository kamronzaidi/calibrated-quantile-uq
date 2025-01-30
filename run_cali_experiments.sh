#!/bin/bash

datasets=('boston' 'yacht' 'energy' 'concrete'  'kin8nm' 'wine' 'power' 'naval' 'protein')
seeds=(0 1 2 3 4)
for data in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        python main.py --seed $seed --data $data --loss scaled_batch_cal_penalty --penalty 0 --gdp 1 --save_dir cali_run
    done
done