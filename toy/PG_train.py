import numpy as np
import random
import torch
from copy import deepcopy
from math import exp, log
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from agent import PolicyGradientAgent
from env import RMDP, get_reward_src, build_toy_env
from utils import print_float_list, print_float_matrix


THRES = 1e-5
T_EST = 100

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Build environment
p_perturb = 0.01
beta  = 3.0
gamma = 0.95

env_name = "Toy-10"
reward_src = get_reward_src(env_name)
env = build_toy_env(reward_src,p_perturb, beta, gamma, THRES)

print("beta", beta, "p_perturb", p_perturb)
print(f"pi_opt = {env.V_to_Q(env.V_opt).argmax(axis=1).flatten().tolist()}.")

M   = 10
eps = 1e-2
eta = 0.1
T   = int(5e4)
print(f"learning rate: {eta}.")

agent = PolicyGradientAgent(env, eta, T_EST, THRES)

pi_init = np.ones(shape=(env.num_states, env.num_actions), dtype=np.float64) / env.num_actions
agent.reset(pi_init)

eval_freq = 20
try:
    loss_list, reward_list = [], []
    bar = tqdm(range(T))
    bar.set_description_str(f"beta = {beta}, p = {p_perturb}, eta = {eta}")
    for t in bar:
        pi, info = agent.update()
        loss_list.append(info["loss"])

        if (t+1) % 20 == 0:
            tqdm.write(f"V_pi: {print_float_list(info['V_pi'])}")
            tqdm.write(f"Q_pi:\n{print_float_matrix(info['Q_pi'].T)}")
            tqdm.write(f"pi:\n{print_float_matrix(pi.T)}")
        bar.set_postfix_str(f"loss: {info['loss']:.6f}")

        if info["loss"] < 1e-6:
            break

        if False: # t % eval_freq == 0:
            test_reps = 10
            test_T = 1000
            cur_rewards = []
            for rep in range(test_reps):
                cur_rewards.append( env.eval(agent.select_action, T_eval=test_T) )
            print(cur_rewards)
            reward_list.append(np.array(cur_rewards, dtype=np.float64).mean())
    
    V_pi = env.DP_pi(pi, THRES)
    loss_list.append( env.V_opt_avg - (V_pi*env.distr_init).sum() )
except KeyboardInterrupt:
    pass

print("#steps", t)
print("V_pi", print_float_list(info["V_pi"]))
print("Q_pi\n", print_float_matrix(info["Q_pi"].T))
print("pi\n", print_float_matrix(pi.T))
print("beta", beta, "p_perturb", p_perturb, "learning rate", eta)
Q_opt = env.V_to_Q(env.V_opt)
print(f"V_opt\n", print_float_matrix(Q_opt.T))
print(f"pi_opt = {Q_opt.argmax(axis=1).flatten().tolist()}.")

# suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
suffix = f"{env_name}_{p_perturb}_{beta}_{eta}"
plt.figure()
plt.plot(np.arange(0, len(loss_list)), loss_list)
plt.xlabel("#iterations")
plt.ylabel(r"$V^{\pi}-loss$")
plt.savefig(f"./log/active/PG_10/{suffix}_loss.png", dpi=200)

np.save(f"./log/active/PG_10/{suffix}_loss.npy", loss_list)
agent.save(f"./log/active/PG_10/{suffix}_Q.npy")

"""plt.figure()
plt.plot(np.arange(0,len(reward_list))*eval_freq, reward_list)
plt.savefig(f"./log/active/PG_10/rewards_{suffix}.png", dpi=200)
np.save(f"./log/active/PG_10/rewards_{suffix}.npy", reward_list)

test_reps = 10
test_T = 1000
reward_list = []
for rep in range(test_reps):
    reward_list.append( env.eval(agent.select_action, T_eval=test_T) )
print(reward_list)
print(np.array(reward_list, dtype=np.float64).mean())"""