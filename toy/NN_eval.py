import argparse
import signal
from tqdm import tqdm
from datetime import datetime
import pickle as pkl

import numpy as np
import torch
import random

from env import CartPole, Pendulum, get_reward_src, build_toy_env
from agent import RFZI_NN
from utils import print_float_list


parser = argparse.ArgumentParser()
parser.add_argument("--agent_prefix", type=str)

parser.add_argument("--seed", default=1024, type=int)
parser.add_argument("--device", default="cpu", type=str, choices=["cpu", "cuda"])

parser.add_argument("--num_eval", default=10, type=int)
parser.add_argument("--T_eval", default=1000, type=int)
parser.add_argument("--thres_eval", default=1e-5, type=float)

parser.add_argument("--disp_V_opt", action="store_true")
parser.add_argument("--disp_V_pi", action="store_true")
parser.add_argument("--disp_policy", action="store_true")

args = parser.parse_args()


# Load settings.
with open(args.agent_prefix + ".pkl", "rb") as f:
    settings = pkl.load(f)

# Logging configuration.
msg  = "="*40 + " Settings " + "="*40 + "\n"
msg += f"agent = RFZI_NN, env = {settings.env}, beta = {settings.beta:.4f}, gamma = {settings.gamma:.4f},\n"
msg += f"p_perturb = {settings.p_perturb}, sigma = {settings.sigma}, dim_emb = {settings.dim_emb},\n"
msg += f"num_eval = {args.num_eval}, T_eval = {args.T_eval}, thres_eval = {args.thres_eval:.4f},\n"
msg += f"disp_V_opt = {args.disp_V_opt}, disp_V_pi = {args.disp_V_pi}, disp_policy = {args.disp_policy}.\n"
msg += "*" * 90 + "\n"
print(msg)


# Random seeding.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
print(f"> Using global random seed {args.seed}.")

# Determine device.
if args.device == "cuda":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    device = torch.device("cpu")
print(f"> Using Pytorch on device {device} ({args.device} requested).")


# Build environment and load data.
is_tabular = False
if settings.env == "CartPole":
    env = CartPole(sigma=settings.sigma)
    
    def emb_func(state):
        return torch.cat([state, torch.sin(state[:,2][:,None]), torch.cos(state[:,2][:,None])], 1)
    dim_emb = env.dim_state + 2
    dim_hidden = (256*env.dim_state, 32)

    print(f"> Setting up CartPole with Gausian noise (sigma = {settings.sigma:.4f}).")
elif settings.env == "Pendulum":
    env = Pendulum(num_actions=args.num_actions, sigma=settings.sigma)
    
    def emb_func(state):
        return state
    dim_emb = env.dim_state
    dim_hidden = (256*env.dim_state, 32)

    print(f"> Setting up Pendulum with Gausian noise (sigma = {settings.sigma:.4f}).")
    print(f"  + Action space contains {args.num_actions} actions: {env.actions}")
elif args.env in ["Toy-10", "Toy-100_design", "Toy-100_Fourier", "Toy-1000"]:
    is_tabular = True

    reward_src = get_reward_src(args.env)
    env = build_toy_env(reward_src, args.p_perturb, args.beta, args.gamma, args.thres_eval, args.disp_V_opt)
    
    pos = args.env.find("_")
    if args.data_path is None: data_path = f"./data/Toy/{args.env}_torch_random.pkl"

    mat = torch.FloatTensor(np.arange(env.num_states)[:, None])
    mat = mat * torch.FloatTensor(np.arange(1, args.dim_emb+1))[None, :]
    mat = mat * (2*torch.pi/env.num_states)
    embedding = torch.cat([torch.sin(mat), torch.cos(mat)], dim=1).to(device)
    def emb_func(state):
        return embedding[state.long().flatten()]
    dim_emb = 2 * args.dim_emb
    dim_hidden = (256*env.dim_state, 32)
    assert dim_emb == len(emb_func(torch.zeros(size=(env.dim_state,))).flatten())

    print(f"> Setting up Toy-10 with stochastic transition (p_perturb = {args.p_perturb:.4f}).")
else:
    raise NotImplementedError

# Display optimal value (only valid for tabular case).
if is_tabular and args.disp_V_opt:
    opt_val = (env.V_opt*env.distr_init).sum()
    msg  = "> Optimal policy for :\n"
    msg += f"  + V_opt = {print_float_list(env.V_opt)}, E[V_opt] = {opt_val:.6f}.\n"
    msg += f"  + pi_opt = {env.V_to_Q(env.V_opt).argmax(axis=1).flatten().tolist()}."
    print(msg)


# Load RFZI agent.
agent = RFZI_NN(
    env=env, device=device,
    beta=settings.beta, gamma=settings.gamma, 
    lr=0, tau=0,
    emb_func=emb_func, dim_emb=dim_emb,
    dim_hidden=dim_hidden
)
print(f"> Setting up agent: beta = {settings.beta}, gamma = {settings.gamma}.")

agent.load(args.agent_prefix + ".ckpt")
print(f"  + Agent loaded from <{args.agent_prefix}.ckpt>.\n\n")

print("\n" + "-"*30 + f" evaluation  starts " + "-"*30)
with torch.no_grad():
    # Display current policy (only valid for tabular case).
    if is_tabular and args.disp_policy:
        cur_policy = []
        for state in env.states:
            cur_policy.append(agent.select_action(state))
        print(f"+ policy = {cur_policy}.")
    
    if is_tabular and args.disp_V_pi:
        V_pi = agent.calc_policy_reward()
        print(f"+ E[V_pi] = {V_pi:.6f}.")

    rewards = []
    for t_eval in range(args.num_eval):
        reward = env.eval(agent.select_action, T_eval=args.T_eval)
        rewards.append(reward)
        print(f">>> Evaluation #{t_eval}: reward = {reward:.6f}.")
    print(f"+ In {args.num_eval} evaluations: avg = {np.average(rewards):.6f}, std = {np.std(rewards):.6f}.")
    
print("-"*80 + "\n\n> Evaluation completed.")