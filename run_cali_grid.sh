#!/bin/bash

#datasets=('boston' 'concrete' 'energy' 'kin8nm' 'power' 'wine' 'naval')
#losses=("batch_int" "batch_qr" "scaled_batch_cal")
seeds=(0 1 2 3 4)

#penalty=(0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1)
#penalty=(0 0.25 0.5 0.75 1)
penalty=(0.05 0.1 0.15 0.2)
#gdp=(1 2 3 5 10 30 100)
#gdp=(1 5 10 30 100)
gdp=(2 3)

for p in "${penalty[@]}"; do
    for g in "${gdp[@]}"; do
        for seed in "${seeds[@]}"; do
            python main.py --seed $seed --data boston --loss scaled_batch_cal_penalty --penalty $p --gdp $g --save_dir cali_test
        done
    done
done