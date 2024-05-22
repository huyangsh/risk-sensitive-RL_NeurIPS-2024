import sys
import os
cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cur_path)
from continuous_cartpole import CartPoleEnv
from pendubot import PendubotEnv
from quadrotor import Quadrotor2D
from cart_pendulum import CartPendulumEnv
from pendulum import PendulumEnvV2

from gymnasium import register

register('Pendulum-v2',
         entry_point='envs:PendulumEnvV2',
         max_episode_steps=200,)