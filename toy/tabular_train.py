import numpy as np
import random
from copy import deepcopy
from math import exp, log
import matplotlib.pyplot as plt

from env import RMDP, get_reward_src, build_toy_env
from data import Dataset
from agent import RFZI_Tabular

THRES = 1e-5

seed = 0
random.seed(seed)
np.random.seed(seed)

# Build environment
p_perturb = 0.15
beta  = 0.01
gamma = 0.95

env_name = "Toy-100_design"
reward_src = get_reward_src(env_name)
env = build_toy_env(p_perturb, beta, gamma, THRES)

# Load data.
# dataset = np.load("./data/Toy/toy_large_random.npy")
dataset = Dataset()
dataset.load("./data/Toy/toy_large_random.pkl")

# Build agent.
agent = RFZI_Tabular(env)
Z_init = np.ones(shape=(env.num_states, env.num_actions), dtype=np.float64)
agent.reset(Z_init)

T = 1000
for t in range(T): 
    _, info = agent.update(dataset.data)
    print(f"loss at {t}: {info['loss']:.6f}, diff = {np.linalg.norm(info['diff']):.6f}.")
    
    if (t % 20 == 0):
        print(f"eval at {t}")
        pi = agent.get_policy()
        print("pi", pi)
        
        n_eval = 10
        T_eval = 1000

        reward_list = []
        for rep in range(n_eval):
            reward_tot = env.eval(agent.select_action, T_eval=T_eval)
            reward_list.append(reward_tot)
        print("rewards", reward_list)

        V_pi = env.DP_pi(pi, thres=THRES)
        V_pi_avg = (V_pi*env.distr_init).sum()
        V_loss = env.V_opt_avg - V_pi_avg
        print(f"V-loss = {V_loss}")