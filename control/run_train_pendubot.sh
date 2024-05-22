#!/bin/bash

for SEED in 1 2 3 4; do
  python main.py --use_random_feature --no_reward_exponential --critic_lr 3e-4 --alg rfsac --env Pendubot-v0 --sigma 0.0 --max_timesteps 150000 --rf_num 4096 --seed $SEED
  for R in 1.0 5.0 10.0; do
    python main.py --use_random_feature --no_reward_exponential --robust_feature --robust_radius $R --critic_lr 3e-4 --alg rfsac --env Pendubot-v0 --sigma 0.0 --max_timesteps 150000 --rf_num 4096 --seed $SEED
  done
done


