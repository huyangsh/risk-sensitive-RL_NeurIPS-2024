import argparse
import signal
from tqdm import tqdm
from datetime import datetime
import pickle as pkl

import numpy as np
import torch
import random

from data import TorchDataset
from env import CartPole, Pendulum
from env import get_reward_src, build_toy_env
from agent import RFZI_NN
from utils import Logger, print_float_list


parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=20, type=int)
parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"])

parser.add_argument("--env", type=str, choices=["CartPole", "Pendulum", "Toy-10", "Toy-100_design", "Toy-100_Fourier", "Toy-100_zone", "Toy-100_zone2", "Toy-1000"])
parser.add_argument("--data_path", type=str)
parser.add_argument("--beta", default=0.01, type=float)
parser.add_argument("--gamma", default=0.95, type=float)
parser.add_argument("--p_perturb", default=0.15, type=float)
parser.add_argument("--sigma", default=0.0, type=float)
parser.add_argument("--num_actions", default=5, type=int)

parser.add_argument("--num_train", default=2000, type=int)
parser.add_argument("--num_batches", default=20, type=int)
parser.add_argument("--batch_size", default=10000, type=int)
parser.add_argument("--lr", default=0.5, type=float)
parser.add_argument("--tau", default=0.1, type=float)
parser.add_argument("--dim_emb", default=100, type=int)

parser.add_argument("--freq_eval", default=10, type=int)
parser.add_argument("--num_eval", default=10, type=int)
parser.add_argument("--T_eval", default=1000, type=int)
parser.add_argument("--thres_eval", default=1e-5, type=float)

parser.add_argument("--freq_save", default=0, type=int)

parser.add_argument("--disp_loss", default=True, type=bool)
parser.add_argument("--eval", default=True, type=bool)
parser.add_argument("--disp_V_opt", action="store_true")
parser.add_argument("--disp_V_pi", action="store_true")
parser.add_argument("--disp_policy", action="store_true")

args = parser.parse_args()


# Logging configuration.
log_prefix  = f"./log/active/{args.env}_"
log_prefix += f"{args.p_perturb}_" if args.env.startswith("Toy") else f"{args.sigma}_"
log_prefix += f"{args.num_train}_{args.num_batches}_{args.batch_size}_{args.lr}_{args.tau}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
logger = Logger(prefix=log_prefix, use_tqdm=True)

msg  = "="*40 + " Settings " + "="*40 + "\n"
msg += f"agent = RFZI_NN, env = {args.env}, beta = {args.beta:.4f}, gamma = {args.gamma:.4f},\n"
msg += f"num_train = {args.num_train}, num_batches = {args.num_batches}, batch_size = {args.batch_size}, lr = {args.lr:.4f}, tau = {args.tau:.4f}, dim_emb = {args.dim_emb},\n"
msg += f"freq_eval = {args.freq_eval}, num_eval = {args.num_eval}, T_eval = {args.T_eval}, thres_eval = {args.thres_eval:.4f},\n"
msg += f"disp_loss = {args.disp_loss}, disp_V_opt = {args.disp_V_opt}, disp_V_pi = {args.disp_V_pi}, disp_policy = {args.disp_policy}, eval = {args.eval}.\n"
msg += "=" * 90 + "\n"
msg += f"> Saving to <{log_prefix}>."
logger.log(msg)

with open(log_prefix + ".pkl", "wb") as f:
    pkl.dump(args, f)
    msg += f"> Settings saved to <{log_prefix}.pkl>.\n"


# Random seeding.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
logger.log(f"> Using global random seed {args.seed}.")

# Determine device.
if args.device == "cuda":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    device = torch.device("cpu")
logger.log(f"> Using Pytorch on device {device} ({args.device} requested).")


# Build environment and load data.
data_path, is_tabular = args.data_path, False
if args.env == "CartPole":
    env = CartPole(sigma=args.sigma)
    if args.data_path is None: data_path = f"./data/CartPole/CartPole_0.01_random.pkl"
    
    def emb_func(state):
        return torch.cat([state, torch.sin(state[:,2][:,None]), torch.cos(state[:,2][:,None])], 1)
    dim_emb = env.dim_state + 2
    dim_hidden = (256*env.dim_state, 32)
    assert dim_emb == len(emb_func(torch.zeros(size=(1,env.dim_state))).flatten())

    logger.log(f"> Setting up CartPole with Gausian noise (sigma = {args.sigma:.4f}).")
    logger.log(f"  + Using data from path <{data_path}>.")
elif args.env == "Pendulum":
    env = Pendulum(num_actions=args.num_actions, sigma=args.sigma)
    if args.data_path is None: data_path = f"./data/Pendulum/Pendulum_random.pkl"
    
    def emb_func(state):
        return state
    dim_emb = env.dim_state 
    dim_hidden = (256*env.dim_state, 32)
    assert dim_emb == len(emb_func(torch.zeros(size=(env.dim_state,))).flatten())

    logger.log(f"> Setting up Pendulum with Gausian noise (sigma = {args.sigma:.4f}).")
    logger.log(f"  + Action space contains {args.num_actions} actions: {env.actions}")
    logger.log(f"  + Using data from path <{data_path}>.")
elif args.env in ["Toy-10", "Toy-100_design", "Toy-100_Fourier", "Toy-100_zone", "Toy-100_zone2", "Toy-1000"]:
    is_tabular = True

    reward_src = get_reward_src(args.env)
    env = build_toy_env(reward_src, args.p_perturb, args.beta, args.gamma, args.thres_eval, args.disp_V_opt)
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

    logger.log(f"> Setting up Toy-10 with stochastic transition (p_perturb = {args.p_perturb:.4f}).")
    logger.log(f"  + Using reward_src vector <{print_float_list(reward_src)}>.")
    logger.log(f"  + Using data from path <{data_path}>.")
else:
    raise NotImplementedError

dataset = TorchDataset(device)
dataset.load(data_path)
logger.log(f"  + Data successfully loaded.")

# Display optimal value (only valid for tabular case).
if is_tabular and args.disp_V_opt:
    opt_val = (env.V_opt*env.distr_init).sum()
    msg  = "> Optimal policy calculated for the tabular environment:\n"
    msg += f"  + V_opt = {print_float_list(env.V_opt)}.\n"
    msg += f"  + E[V_opt] = {opt_val:.6f}.\n"
    msg += f"  + pi_opt = {env.V_to_Q(env.V_opt).argmax(axis=1).flatten().tolist()}."
    logger.log(msg)

# Train RFZI agent.
agent = RFZI_NN(
    env=env, device=device,
    beta=args.beta, gamma=args.gamma, 
    lr=args.lr, tau=args.tau,
    emb_func=emb_func, dim_emb=dim_emb,
    dim_hidden=dim_hidden
)
logger.log(f"> Setting up agent: beta = {args.beta}, gamma = {args.gamma}, lr = {args.lr}, tau = {args.tau}.\n\n")
if args.freq_save > 0:
    logger.log(f"  + Automatically save agent every {args.freq_save} iterations.")

try:
    for t in tqdm(range(args.num_train)):
        # Update agent.
        info = agent.update(dataset, num_batches=args.num_batches, batch_size=args.batch_size)
        if args.disp_loss:
            logger.log(f"Iteration #{t+1}: losses = {print_float_list(info['loss'])}.")
        else:
            logger.log(f"Iteration #{t+1}: finished.")
        
        # Automatic saving.
        if args.freq_save > 0 and (t+1) % args.freq_save == 0:
            agent.save(f"{log_prefix}_{t+1}.ckpt")
            logger.log(f"\n> Current agent automatically saved to <{log_prefix}_{t+1}.ckpt>.\n")
        
        # Periodic evaluation.
        if (t+1) % args.freq_eval == 0:
            logger.log("\n" + "-"*30 + f" evaluate at iteration # {str(t+1).rjust(4)} " + "-"*30)

            with torch.no_grad():
                # Display current policy (only valid for tabular case).
                if is_tabular and args.disp_policy:
                    cur_policy = []
                    for state in env.states:
                        cur_policy.append(agent.select_action(state))
                    logger.log(f"+ policy = {cur_policy}.")

                if args.eval:
                    rewards = []
                    for t_eval in range(args.num_eval):
                        reward = env.eval(agent.select_action, T_eval=args.T_eval)
                        rewards.append(reward)
                    logger.log(f"+ episodic rewards = {print_float_list(rewards)}.")
                    logger.log(f"+ average reward = {np.average(rewards):.6f}, std = {np.std(rewards):.6f}.")
                
                if is_tabular and args.disp_V_pi:
                    V_pi = agent.calc_policy_reward()
                    logger.log(f"+ E[V_pi] = {V_pi:.6f}.")
            
            logger.log("-"*90 + "\n")
except KeyboardInterrupt:
    pass

agent.save(log_prefix + ".ckpt")
logger.log(f"\n\n> Training completed after {t} iterations: agent saved to <{log_prefix}.ckpt>.")
logger.save()