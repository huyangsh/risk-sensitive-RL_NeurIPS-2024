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

pi_opt = list(env.V_to_Q(env.V_opt).argmax(axis=1))
print(pi_opt)

prefix = f"./log/selected/{args.env}_{args.beta}/{args.env}_{args.beta}"
seeds = [0, 10, 20, 30, 40]

pis = [pi_std, pi_opt]
for i in range(5):
    seed = seeds[i]
    agent.load(f"{prefix}_{seed}.ckpt")

    pi = []
    for s in env.states:
        pi.append(agent.select_action(np.array([s])))
    print(pi)
    pis.append(pi)


reward_std, reward_opt, reward_agent = [], [], []
alpha_list = np.arange(0, 0.5, 0.01)
for alpha in tqdm(alpha_list):
    values = policy_eval_alpha(alpha, pis)
    print(values)

    reward_std.append(values[0])
    reward_opt.append(values[1])
    reward_agent.append(values[2:])

with open(f"./plot/{args.env}_{args.beta}_alpha_{args.mode}.pkl" ,"wb") as f:
    pkl.dump([reward_std, reward_opt, reward_agent], f)


reward_agent = np.array(reward_agent).T
reward_agent_avg = reward_agent.mean(axis=0)
reward_agent_std = reward_agent.std(axis=0)

reward_std = np.array(reward_std)
reward_opt = np.array(reward_opt)

plt.plot(alpha_list, reward_std, label="classical")
plt.plot(alpha_list, reward_opt, label="optimal")
plt.plot(alpha_list, reward_agent_avg, label="RFZI")
plt.fill_between(alpha_list, reward_agent_avg-reward_agent_std, reward_agent_avg+reward_agent_std, color="C1", alpha=0.1)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"value at $\alpha$")
plt.legend()
plt.savefig(f"./plot/{args.env}_{args.beta}_alpha_{args.mode}.png", dpi=200)