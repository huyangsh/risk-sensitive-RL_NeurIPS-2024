#!/bin/bash

for SEED in 1 2 3 4; do
  python main.py --use_random_feature --critic_lr 3e-4 --alg $ALG --env Pendulum-v1 --sigma 0.0 --max_timesteps 80000 --rf_num 512 --seed $SEED
  for R in 0.5 2.0 5.0; do
    python main.py --use_random_feature --robust_feature --robust_radius $R --critic_lr 3e-4 --alg $ALG --env Pendulum-v1 --sigma 0.0 --max_timesteps 80000 --rf_num 512 --seed $SEED
  done
done

