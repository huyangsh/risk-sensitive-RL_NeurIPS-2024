import numpy as np
import scipy as sp
import torch
from math import sin, cos, pi
import argparse
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm

from agent import RFZI_NN
from env import get_reward_src, build_toy_env


T = 100
print(T)


parser = argparse.ArgumentParser()

parser.add_argument("--mode", default="discounted", type=str, choices=["cumulative", "discounted"])
parser.add_argument("--seed", default=20, type=int)
parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"])

parser.add_argument("--env", default="Toy-100_zone", type=str, choices=["Toy-10", "Toy-100_design", "Toy-100_Fourier", "Toy-100_zone", "Toy-1000"])
parser.add_argument("--data_path", type=str)
parser.add_argument("--beta", default=0.5, type=float)
parser.add_argument("--gamma", default=0.95, type=float)
parser.add_argument("--p_perturb", default=0.15, type=float)
parser.add_argument("--sigma", default=0.0, type=float)
parser.add_argument("--num_actions", default=5, type=int)

parser.add_argument("--lr", default=0.5, type=float)
parser.add_argument("--tau", default=0.1, type=float)
parser.add_argument("--dim_emb", default=100, type=int)
parser.add_argument("--thres_eval", default=1e-5, type=float)

args = parser.parse_args()

if args.device == "cuda":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    device = torch.device("cpu")


if args.mode == "cumulative":
    gamma = 1.00
elif args.mode == "discounted":
    gamma = args.gamma
else:
    raise NotImplementedError

if args.env in ["Toy-10", "Toy-100_design", "Toy-100_Fourier", "Toy-100_zone", "Toy-1000"]:
    is_tabular = True
    reward_src = get_reward_src(args.env)
    print(reward_src)
    env = build_toy_env(reward_src, args.p_perturb, args.beta, args.gamma, args.thres_eval, True)

    mat = torch.FloatTensor(np.arange(env.num_states)[:, None])
    mat = mat * torch.FloatTensor(np.arange(1, args.dim_emb+1))[None, :]
    mat = mat * (2*torch.pi/env.num_states)
    embedding = torch.cat([torch.sin(mat), torch.cos(mat)], dim=1).to(device)
    def emb_func(state):
        return embedding[state.long().flatten()]
    dim_emb = 2 * args.dim_emb
    dim_hidden = (256*env.dim_state, 32)
    assert dim_emb == len(emb_func(torch.zeros(size=(env.dim_state,))).flatten())


agent = agent = RFZI_NN(
    env=env, device=device,
    beta=args.beta, gamma=gamma, 
    lr=args.lr, tau=args.tau,
    emb_func=emb_func, dim_emb=dim_emb,
    dim_hidden=dim_hidden
)

delta_list = np.arange(0.32, 0.51, 0.02)


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

def policy_eval_robust(env, reward_src, p_perturb, T, pi, delta):
    V_robust = np.zeros([T, env.num_states])
    V_robust[T-1, :] = reward_src
    for t in range(T-2, -1, -1):
        for s in range(env.num_states):
            a = pi[s]
            s_r = (s+a) % env.num_states
            s_l = (s+a-2) % env.num_states
            s_p = (s+a-1) % env.num_states
            V_next = np.array([V_robust[t+1, s_l], V_robust[t+1, s_p], V_robust[t+1, s_r]])
            mu_0 = np.array([p_perturb, 1-2*p_perturb, p_perturb])

            constraint_1 = sp.optimize.NonlinearConstraint(lambda x: sp.special.rel_entr(np.array(x), mu_0).sum(), lb=-np.inf, ub=delta)
            constraint_2 = sp.optimize.LinearConstraint(np.ones(shape=(3,)), lb=1, ub=1)
            sol = sp.optimize.minimize(
                fun = lambda x: np.dot(np.array(x), V_next),
                x0 =  mu_0,
                constraints = [constraint_1, constraint_2]
            )

            V_robust[t, s] = reward_src[s] + gamma * sol.fun

    return np.dot(env.distr_init, V_robust[0, :])

pi_std = list(V_to_Q(env, DP_opt_std(env)).argmax(axis=1))
pi_opt = list(env.V_to_Q(env.V_opt).argmax(axis=1))
print(pi_std)
print(pi_opt)

reward_std, reward_opt = [], []
for delta in tqdm(delta_list):
    r_s = policy_eval_robust(env, reward_src, args.p_perturb, T, pi_std, delta=delta)
    r_o = policy_eval_robust(env, reward_src, args.p_perturb, T, pi_opt, delta=delta)

    print(delta, r_s, r_o)
    reward_std.append(r_s)
    reward_opt.append(r_o)

prefix = f"./log/selected/{args.env}_{args.beta}/{args.env}_{args.beta}"
seeds = [0, 10, 20, 30, 40]
reward_agent = [[], [], [], [], []]

for i in range(5):
    seed = seeds[i]
    agent.load(f"{prefix}_{seed}.ckpt")

    pi = []
    for s in env.states:
        pi.append(agent.select_action(np.array([s])))
    print(pi)

    for delta in tqdm(delta_list):
        r_a = policy_eval_robust(env, reward_src, args.p_perturb, T, pi, delta=delta)

        print(delta, r_a)
        reward_agent[i].append(r_a)
print(reward_agent)

with open(f"./plot/{args.env}_{args.beta}_robust_{T}.pkl" ,"wb") as f:
    pkl.dump([reward_agent, reward_std, reward_opt], f)

reward_agent = np.array(reward_agent)
reward_agent_avg = reward_agent.mean(axis=0)
reward_agent_std = reward_agent.std(axis=0)

reward_std = np.array(reward_std)
reward_opt = np.array(reward_opt)

plt.plot(delta_list, reward_std, label="classical")
plt.plot(delta_list, reward_opt, label="optimal")
plt.plot(delta_list, reward_agent_avg, label="RFZI")
plt.fill_between(delta_list, reward_agent_avg-reward_agent_std, reward_agent_avg+reward_agent_std, color="C2", alpha=0.1)
plt.xlabel(r"$\delta$")
plt.ylabel(r"$\hat{V}_{\pi}(\delta)$")
plt.legend()
plt.savefig(f"./plot/{args.env}_{args.beta}_robust_{T}.png", dpi=200)