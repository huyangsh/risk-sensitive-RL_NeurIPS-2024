import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import pickle as pkl

from agent import EpisodicPolicyGradientAgent
from env import LinearEpisodicMDP, get_linear_param
from utils import print_float_list, print_episodic_matrix


THRES = 1e-5
T_EST = 100
T_Q   = 20   # 20 for SDP, 100 for SLSQP.

seed = 0
random.seed(seed)
np.random.seed(seed)
# torch.manual_seed(seed)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

H         = 20
EPS_PHI   = 0.20
EPS_THETA = 0
EPS_MU    = 0.1
USE_SDP   = True

alpha     = 0.2
num_iter  = 100
print(f"learning rate: {alpha}.")


# Build environment
env_name = "Toy-4"
distr_init, phi, theta, mu, mu_perturb = get_linear_param(env_name)
env = LinearEpisodicMDP(distr_init, phi, theta, mu, H, eps_phi=EPS_PHI, eps_theta=EPS_THETA, eps_mu=EPS_MU, thres=1e-5)
test_env = LinearEpisodicMDP(distr_init, phi, theta, mu_perturb, H, thres=1e-5)

# Reference policies.
pi_nominal = env.get_opt_pi()
pi_perturb = test_env.get_opt_pi()
reward_nominal = test_env.distr_init @ test_env.DP_pi(pi=pi_nominal)[0,:]
reward_perturb = test_env.distr_init @ test_env.DP_pi(pi=pi_perturb)[0,:]

# tqdm.write(f"pi_nominal:\n{print_episodic_matrix(pi_nominal)}\nreward = {reward_nominal}.\n")
# tqdm.write(f"pi_perturb:\n{print_episodic_matrix(pi_perturb)}\nreward = {reward_perturb}.")

agent = EpisodicPolicyGradientAgent(env, alpha, H)

pi_init = np.ones(shape=(H, env.num_states, env.num_actions), dtype=np.float32) / env.num_actions
agent.reset(pi_init)

eval_freq = 1
try:
    reward_list, pi_list, t_list = [], [], []
    bar = tqdm(range(num_iter))
    # bar.set_description_str(f"alpha = {alpha}, eps = ({EPS_PHI}, {EPS_THETA}, {EPS_MU}), target_reward = [{reward_nominal:.4g}, {reward_perturb:.4g}]")
    bar.set_description_str(f"alpha = {alpha}, eps = ({EPS_PHI}, const {EPS_MU}), target_reward = [{reward_nominal:.4g}, {reward_perturb:.4g}]")
    for t in bar:
        pi, info = agent.update()
        pi_list.append(pi)

        if t % eval_freq == 0:
            tqdm.write(f"Evaluate at iteration #{t}:")
            # tqdm.write(f"Q_pi:\n{print_float_matrix(info['Q_pi'].T)}")
            # tqdm.write(f"pi:\n{print_episodic_matrix(pi[:5,:,:])}")
            tqdm.write(f"V_pi:\n{print_float_list(env.Q_to_V(info['Q_pi'],pi)[0,:])}")
            t_list.append(t)

            test_reps = 10
            test_T = 1000
            cur_rewards = []
            for rep in range(test_reps):
                cur_rewards.append( test_env.episodic_eval(agent.select_action) )
            tqdm.write(f"\ntest rewards:\n{print_float_list(cur_rewards)}\n")

            V_pi = test_env.DP_pi(pi=pi)
            avg_reward = test_env.distr_init @ V_pi[0,:]
            tqdm.write(f"average reward: {avg_reward}\n\n")
            reward_list.append(avg_reward)
except KeyboardInterrupt:
    pass

# with open(f"./log/{timestamp}_{H}_{EPS_PHI:.3f}_{EPS_THETA:.3f}_{EPS_MU:.3f}_{alpha:.2f}.pkl", "ab") as f:
with open(f"./log/{timestamp}_{H}_{EPS_PHI:.3f}_const_{EPS_MU:.3f}_{alpha:.2f}.pkl", "ab") as f:
    pkl.dump({
        "t": t_list,
        "pi": pi_list,
        "reward": reward_list
    }, f)
print("Successfully saved.")
exit()

# Plot rewards.
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.plot(t_list, reward_list, label=r"$\varepsilon=$" + f"{EPS_PHI, EPS_THETA, EPS_MU}")
ax.axhline(y=reward_nominal, linestyle="--", color="r", label="nominal optimal")
ax.axhline(y=reward_perturb, linestyle="--", color="g", label="perturbed optimal")
ax.legend(loc="lower right")
fig.savefig(f"./log/linear_PG_{env_name}_{timestamp}_reward_{EPS_PHI:.3f}_{EPS_THETA:.3f}_{EPS_MU:.3f}_{alpha:.2f}.png", dpi=300)

# Plot policy changes over time.
if env_name == "Toy-4":
    fig = plt.figure(figsize=(20,5))
    t_horizon = list(range(len(pi_list)))
    for i in range(4):
        ax = fig.add_subplot(1,4,i+1)
        ax.plot(t_horizon, [pi[i,0] for pi in pi_list], label="left")
        ax.plot(t_horizon, [pi[i,1] for pi in pi_list], label="stay")
        ax.plot(t_horizon, [pi[i,2] for pi in pi_list], label="right")
        ax.legend()
        ax.set_xlabel(f"State {i}")
        ax.set_ylabel(f"probability")
    fig.savefig(f"./log/linear_PG_{env_name}_{timestamp}_policy_{EPS_PHI:.3f}_{EPS_THETA:.3f}_{EPS_MU:.3f}_{alpha:.2f}.png", dpi=300)
elif env_name == "Mixture":
    fig = plt.figure(figsize=(20,5))
    t_horizon = list(range(len(pi_list)))
    for i in range(3):
        ax = fig.add_subplot(1,3,i+1)
        ax.plot(t_horizon, [pi[i,0] for pi in pi_list], label="high risk")
        ax.plot(t_horizon, [pi[i,1] for pi in pi_list], label="low risk")
        ax.legend()
        ax.set_xlabel(f"State {i}")
        ax.set_ylabel(f"probability")
    fig.savefig(f"./log/linear_PG_{env_name}_{timestamp}_policy_{EPS_PHI:.3f}_{EPS_THETA:.3f}_{EPS_MU:.3f}_{alpha:.2f}.png", dpi=300)
else:
    print("Policy plotting error: not implemented.")