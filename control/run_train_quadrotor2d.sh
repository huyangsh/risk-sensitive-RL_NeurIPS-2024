#!/bin/bash

for SEED in 1 2 3 4; do # --critic_lr 1e-3
  python main.py --use_random_feature --device cuda --reward_exponential --learn_rf --critic_lr 1e-3 --alg rfsac --env Quadrotor2D-v2 --sigma 0.0 --max_timesteps 150000 --rf_num 4096 --seed $SEED
  for R in 1.0 5.0 10.0; do
    python main.py --use_random_feature --robust_feature --robust_radius $R --device cuda --reward_exponential --learn_rf --critic_lr 1e-3 --alg rfsac --env Quadrotor2D-v2 --sigma 0.0 --max_timesteps 150000 --rf_num 4096 --seed $SEED
  done
done