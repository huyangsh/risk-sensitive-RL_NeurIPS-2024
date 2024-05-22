#!/bin/bash

for SEED in 1 2 3 4; do
    python main.py --use_random_feature --device cuda  --critic_lr 3e-4 --alg rfsac --env CartPendulum-v0 --sigma 0.0 --max_timesteps 150000 --rf_num 8192 --seed $SEED
    for R in 5.0 10.0 20.0; do
      python main.py --use_random_feature --device cuda --robust_feature --robust_radius $R --critic_lr 3e-4 --alg rfsac --env CartPendulum-v0 --sigma 0.0 --max_timesteps 150000 --rf_num 8192 --seed $SEED
    done
done


