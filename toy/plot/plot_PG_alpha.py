import numpy as np
import scipy as sp
import torch
from math import sin, cos, pi
import argparse
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm

from env import get_reward_src, build_toy_env

parser = argparse.ArgumentParser()

parser.add_argument("--mode", default="discounted", type=str, choices=["cumulative", "discounted"])
parser.add_argument("--seed", default=20, type=int)
parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"])

parser.add_argument("--env", default="Toy-10", type=str, choices=["Toy-10", "Toy-100_design", "Toy-100_Fourier", "Toy-1000"])
parser.add_argument("--data_path", type=str)
parser.add_argument("--beta", default=0.01, type=float)
parser.add_argument("--gamma", default=0.95, type=float)
parser.add_argument("--p_perturb", default=0.15, type=float)
parser.add_argument("--sigma", default=0.0, type=float)
parser.add_argument("--dim_emb", default=10, type=int)
parser.add_argument("--thres_eval", default=1e-5, type=float)

args = parser.parse_args()

if args.device == "cuda":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    device = torch.device("cpu")

gamma = args.gamma

reward_src = get_reward_src(args.env)
print(args.p_perturb, reward_src)
env = build_toy_env(reward_src, args.p_perturb, args.beta, args.gamma, args.thres_eval, True)


def DP_opt_std(env, thres=1e-5):
    V = np.zeros(shape=(env.num_states,), dtype=np.float64)

    diff = thres + 1
    while diff > thres:
        V_prev = V
        V = np.zeros(shape=(env.num_states,), dtype=np.float64)

        for s in env.states:
            reward_max = None
            for a in env.actions:
                V_pi_cum = 0
                for s_ in env.states:
                    V_pi_cum += env.prob[s,a,s_] * V_prev[s_]

                if reward_max is None:
                    reward_max = env.reward[s,a] + env.gamma*V_pi_cum
                else:
                    reward_max = max(reward_max, env.reward[s,a] + env.gamma*V_pi_cum)
            
            V[s] = reward_max
        
        diff = np.linalg.norm(V - V_prev)
    
    return V

def V_to_Q(env, V):
    assert V.shape == (env.num_states,)
    Q = np.zeros(shape=(env.num_states, env.num_actions), dtype=np.float64)
    for s in env.states:
        for a in env.actions:
            V_pi_cum = 0
            for s_ in env.states:
                V_pi_cum += env.prob[s,a,s_] * V[s_]

            Q[s,a] = env.reward[s,a] + env.gamma*V_pi_cum
    
    return Q


def DP_pi_std(env, pi, thres=1e-5):
    V = np.zeros(shape=(env.num_states,), dtype=np.float64)

    diff = thres + 1
    while diff > thres:
        V_prev = V
        V = np.zeros(shape=(env.num_states,), dtype=np.float64)

        for s in env.states:
            a = pi[s]

            V_pi_cum = 0
            for s_ in env.states:
                V_pi_cum += env.prob[s,a,s_] * V_prev[s_]
            
            V[s] = env.reward[s,a] + env.gamma*V_pi_cum
        
        diff = np.linalg.norm(V - V_prev)
    
    return V

def policy_eval_alpha(alpha, policies):
    eval_env = build_toy_env(reward_src, alpha, args.beta, args.gamma, args.thres_eval, False)
    values = []
    for policy in policies:
        values.append( np.dot(eval_env.distr_init, DP_pi_std(eval_env, policy, 1e-5)) )
    return values


pi_std = list(V_to_Q(env, DP_opt_std(env)).argmax(axis=1))
print(pi_std)

"""
pis = [ pi_std,
        [2, 2, 2, 1, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2],
        [0, 0, 2, 1, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0] ]
labels = [r"policy gradient ($\beta = 1.0$)", r"policy gradient ($\beta = 2.0$)", r"policy gradient ($\beta = 3.0$)"]
tag = "0.01"
"""
pis = [ [2, 2, 2, 1, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0] ]
labels = [r"policy gradient ($beta = 0.1$)", r"policy gradient ($beta = 1.0$)"]
tag = "0.15"


reward_std, reward_pis = [], []
alpha_list = np.arange(0, 0.5, 0.01)
for alpha in tqdm(alpha_list):
    values = policy_eval_alpha(alpha, pis)
    print(values)

    reward_std.append(values[0])
    reward_pis.append(values[1:])

with open(f"./plot/PG_{tag}_{args.env}_alpha_{args.mode}.pkl" ,"wb") as f:
    pkl.dump([reward_pis, reward_std], f)

reward_pis = np.array(reward_pis)
reward_std = np.array(reward_std)

plt.plot(alpha_list, reward_std, label="classical")
plt.plot(alpha_list, reward_pis, label=labels)
plt.xlabel(r"$\delta$")
plt.ylabel(r"$\hat{V}_{\pi}(\delta)$")
plt.legend()
plt.savefig(f"./plot/PG_{tag}_{args.env}_alpha_{args.mode}.png", dpi=200)