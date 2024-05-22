import argparse
import signal
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime
import pickle as pkl

import numpy as np
import torch
import random

from data import TorchBuffer
from env import CartPole, Pendulum
from env import get_reward_src, build_toy_env
from agent import RFZI_NN
from utils import Logger, print_float_list


parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=20, type=int)
parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"])

parser.add_argument("--env", type=str, choices=["CartPole", "Pendulum", "Toy-10", "Toy-100", "Toy-1000"])
parser.add_argument("--data_path", type=str)
parser.add_argument("--beta", default=0.01, type=float)
parser.add_argument("--gamma", default=0.95, type=float)
parser.add_argument("--p_perturb", default=0.15, type=float)
parser.add_argument("--sigma", default=0.0, type=float)
parser.add_argument("--num_actions", default=5, type=int)

parser.add_argument("--T_train", default=100000, type=int)
parser.add_argument("--batch_size", default=10000, type=int)
parser.add_argument("--lr", default=0.5, type=float)
parser.add_argument("--tau", default=0.1, type=float)
parser.add_argument("--dim_emb", default=100, type=int)
parser.add_argument("--freq_update", default=1, type=int)

parser.add_argument("--eps", default=0.01, type=float)
parser.add_argument("--buffer_size", default=1000000, type=int)
parser.add_argument("--off_ratio", default=0.1, type=float)

parser.add_argument("--freq_eval", default=1000, type=int)
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
log_prefix += f"{args.T_train}_{args.batch_size}_{args.lr}_{args.tau}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
logger = Logger(prefix=log_prefix, use_tqdm=True, flush_freq=1)

msg  = "="*40 + " Settings " + "="*40 + "\n"
msg += f"agent = RFZI_NN, env = {args.env}, beta = {args.beta:.4f}, gamma = {args.gamma:.4f},\n"
msg += f"T_train = {args.T_train}, batch_size = {args.batch_size}, lr = {args.lr:.4f}, tau = {args.tau:.4f}, dim_emb = {args.dim_emb}, freq_update = {args.freq_update},\n"
msg += f"eps = {args.eps}, buffer_size = {args.buffer_size}, offline_data_ratio = {args.off_ratio:.4f},\n"
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
    env_train = CartPole(sigma=args.sigma)
    if args.data_path is None: data_path = f"./data/CartPole/CartPole_0.01_random.pkl"
    
    def emb_func(state):
        return torch.cat([state, torch.sin(state[:,2][:,None]), torch.cos(state[:,2][:,None])], 1)
    dim_emb = env_train.dim_state + 2
    assert dim_emb == len(emb_func(torch.zeros(size=(1,env_train.dim_state))).flatten())

    logger.log(f"> Setting up CartPole with Gausian noise (sigma = {args.sigma:.4f}).")
    logger.log(f"  + Using data from path <{data_path}>.")
elif args.env == "Pendulum":
    env_train = Pendulum(num_actions=args.num_actions, sigma=args.sigma)
    if args.data_path is None: data_path = f"./data/Pendulum/Pendulum_random.pkl"
    
    def emb_func(state):
        return state
    dim_emb = env_train.dim_state 
    assert dim_emb == len(emb_func(torch.zeros(size=(env_train.dim_state,))).flatten())

    logger.log(f"> Setting up Pendulum with Gausian noise (sigma = {args.sigma:.4f}).")
    logger.log(f"  + Action space contains {args.num_actions} actions: {env_train.actions}")
    logger.log(f"  + Using data from path <{data_path}>.")
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

    logger.log(f"> Setting up Toy-10 with stochastic transition (p_perturb = {args.p_perturb:.4f}).")
    logger.log(f"  + Using reward_src vector <{print_float_list(reward_src)}>.")
    logger.log(f"  + Using data from path <{data_path}>.")
else:
    raise NotImplementedError

env_eval = deepcopy(env_train)
buffer = TorchBuffer(device, env_train.dim_state, env_train.dim_action, args.buffer_size, args.off_ratio)
buffer.load(data_path, shuffle=False)
logger.log(f"  + Data successfully loaded.")

# Display optimal value (only valid for tabular case).
if is_tabular and args.disp_V_opt:
    opt_val = (env_train.V_opt*env_train.distr_init).sum()
    msg  = "> Optimal policy calculated for the tabular environment:\n"
    msg += f"  + V_opt = {print_float_list(env_train.V_opt)}.\n"
    msg += f"  + E[V_opt] = {opt_val:.6f}.\n"
    msg += f"  + pi_opt = {env_train.V_to_Q(env_train.V_opt).argmax(axis=1).flatten().tolist()}."
    logger.log(msg)

# Train RFZI agent.
agent = RFZI_NN(
    env=env_train, device=device,
    beta=args.beta, gamma=args.gamma, 
    lr=args.lr, tau=args.tau,
    emb_func=emb_func, dim_emb=dim_emb,
    dim_hidden=(256*env_train.dim_state, 32),
    auto_transfer=False
)
logger.log(f"> Setting up agent: beta = {args.beta}, gamma = {args.gamma}, lr = {args.lr}, tau = {args.tau}.\n\n")
if args.freq_save > 0:
    logger.log(f"  + Automatically save agent every {args.freq_save} iterations.")

try:
    episode = 0
    step_cnt, reward_tot, loss_list = 0, 0, []
    state, done = env_train.reset(), False
    for t in tqdm(range(args.T_train)):
        # Take one step and collect data.
        if np.random.binomial(n=1, p=args.eps):
            action = random.choice(env_train.actions)
        else:
            action = agent.select_action(state)
        next_state, reward, done, _ = env_train.step(action)
        buffer.add(state, action, reward, next_state, done)
        
        state       = next_state
        reward_tot += reward
        step_cnt   += 1
        
        if done:  # Restart at the end of episode.
            episode += 1
            logger.log(f"Episode #{episode}: {step_cnt} steps, cumulative reward = {reward_tot:.4f}.")
            logger.log(f"  Z_func loss: {print_float_list(loss_list)}.")

            state, done = env_train.reset(), False
            step_cnt, reward_tot, loss_list = 0, 0, []

        # Update agent.
        info = agent.update(buffer, num_batches=1, batch_size=args.batch_size)
        if args.disp_loss:
            loss_list.append(info["loss"][0])
        if (t+1) % args.freq_update == 0:
            agent.update_target()
        
        # Automatic saving.
        if args.freq_save > 0 and (t+1) % args.freq_save == 0:
            agent.save(f"{log_prefix}_{t+1}.ckpt")
            logger.log(f"\n> Current agent automatically saved to <{log_prefix}_{t+1}.ckpt>.\n")
        
        # Periodic evaluation.
        if (t+1) % args.freq_eval == 0:
            logger.log("\n" + "-"*30 + f" evaluate after step # {str(t+1).rjust(6)} " + "-"*30)

            with torch.no_grad():
                # Display current policy (only valid for tabular case).
                if is_tabular and args.disp_policy:
                    cur_policy = []
                    for state in env_train.states:
                        cur_policy.append(agent.select_action(state))
                    logger.log(f"+ policy = {cur_policy}.")

                if args.eval:
                    rewards = []
                    for t_eval in range(args.num_eval):
                        reward = env_eval.eval(agent.select_action, T_eval=args.T_eval)
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