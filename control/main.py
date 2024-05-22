import numpy as np
import torch
# import gym
import argparse
import os
import json
import pickle as pkl

from tensorboardX import SummaryWriter

from utils import util, buffer
from agent.sac import sac_agent
from agent.vlsac import vlsac_agent
from agent.rfsac import rfsac_agent
from datetime import datetime

# from envs.noisy_pend import noisyPendulumEnv
from envs.env_helper import *

ENV_CONFIG = {'sin_input': True,  # fixed
              'reward_exponential': False,  # fixed
              'reward_scale': 1.,  # further tune
              'reward_type': 'lqr',  # control different envs
              'theta_cal': 'sin_cos',  # fixed
              'noisy': False,  # todo:depreciated
              'noise_scale': 0.  # should be same with sigma
              }

DEVICE = "cuda"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Names and pipelines
    parser.add_argument("--notes", default='adjust_norm', type=str)
    parser.add_argument("--dir", default=1, type=int)
    parser.add_argument("--device", default='cpu', type=str, help='device, either cpu or cuda')
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--eval_freq", default=5000, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=10e4, type=float)  # Max time steps to run environment
    parser.add_argument("--start_timesteps", default=5e3, type=float)  # Time steps initial random policy is used
    parser.add_argument("--alg", default="rfsac")  # Alg name (sac, vlsac)
    parser.add_argument("--sigma", default=0., type=float)  # noise for noisy environment
    parser.add_argument("--embedding_dim", default=-1, type=int)  # if -1, do not add embedding layer

    # envs
    parser.add_argument("--env", default="Pendulum-v1")  # Environment name
    parser.add_argument("--reward_exponential", action='store_true')
    parser.add_argument("--no_reward_exponential", dest='reward_exponential', action='store_false')
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds

    # models
    parser.add_argument("--hidden_dim", default=256, type=int)  # Network hidden dims
    parser.add_argument("--feature_dim", default=256, type=int)  # Latent feature dim

    # ALGORITHMS
    parser.add_argument("--rf_num", default=512, type=int)
    parser.add_argument("--nystrom_sample_dim", default=512, type=int,
                        help='sample dim, must be greater or equal rf num.')
    parser.add_argument("--learn_rf", action='store_true')
    parser.add_argument("--euler",
                        action='store_true')  # True if euler discretization to be used; otherwise use default OpenAI gym discretization
    parser.add_argument("--use_nystrom", action='store_true')
    parser.add_argument("--robust_feature", action='store_true')
    parser.add_argument("--robust_radius", default=1.0, type=float)
    parser.add_argument("--use_random_feature", dest='use_nystrom', action='store_false')
    parser.add_argument("--learn_bonus", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--extra_feature_steps", default=3, type=int)
    parser.set_defaults(use_nystrom=False)
    parser.set_defaults(robust_feature=False)
    parser.set_defaults(euler=False)
    parser.set_defaults(learn_rf=False)  # if want to add these, just add --use_nystrom to the scripts.
    parser.set_defaults(reward_exponential=ENV_CONFIG['reward_exponential'])


    # SAC parameters
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate

    parser.add_argument("--critic_lr", type=float, default=3e-4)

    args = parser.parse_args()

    sigma = args.sigma
    euler = True if args.euler == True else False
    use_nystrom = True if args.use_nystrom == True else False

    ENV_CONFIG.update({'noisy': args.sigma, 'noise_scale': args.sigma})
    for key, item in vars(args).items():
        if key in ENV_CONFIG.keys():
            ENV_CONFIG.update({key: item})

    if args.env == "Pendulum-v1":
        ENV_CONFIG.update({'reward_scale': 0.2, })
        env = env_creator_pendulum(ENV_CONFIG)
        ENV_CONFIG.update({'reward_scale': 1., })
        eval_env = env_creator_pendulum(ENV_CONFIG)
    elif args.env == 'Quadrotor2D-v2':
        eval_config = ENV_CONFIG.copy()
        eval_config.update({'reward_scale': 1., 'eval': True, 'reward_exponential': False})
        eval_env = env_creator_quad2d(eval_config)
        ENV_CONFIG.update({'reward_scale': 10., })
        env = env_creator_quad2d(ENV_CONFIG)
    elif args.env == 'Pendubot-v0':
        eval_config = ENV_CONFIG.copy()
        eval_config.update({'reward_scale': 1., 'eval': True, })  # 'reward_type': 'energy',
        print(eval_config)
        eval_env = env_creator_pendubot(eval_config)
        ENV_CONFIG.update({'reward_scale': 3.})
        env = env_creator_pendubot(ENV_CONFIG)
    elif args.env == 'CartPoleContinuous-v0':
        ENV_CONFIG.update({'reward_scale': 1., 'eval': True, 'reward_exponential': False})
        eval_env = env_creator_cartpole(ENV_CONFIG)
        ENV_CONFIG.update({'reward_scale': 0.3, 'reward_exponential': ENV_CONFIG.get('reward_exponential'), 'eval': False})
        env = env_creator_cartpole(ENV_CONFIG)
    elif args.env == 'CartPendulum-v0':
        ENV_CONFIG.update({'reward_scale': 1., 'eval': True, 'reward_exponential': False})
        eval_env = env_creator_cartpendulum(ENV_CONFIG)
        ENV_CONFIG.update({'reward_scale': 0.3, 'reward_exponential': ENV_CONFIG.get('reward_exponential'), 'eval': False})
        env = env_creator_cartpendulum(ENV_CONFIG)

    # wrapper back to gym to fit the code
    env = Gymnasium2GymWrapper(env)
    eval_env = Gymnasium2GymWrapper(eval_env)

    env_name = f'{args.env}_sigma_{args.sigma}_rew_scale_{ENV_CONFIG["reward_scale"]}'

    if args.env == 'Pendubot-v0':
        env_name = env_name + f'_reward_{ENV_CONFIG["reward_type"]}'

    if args.alg == 'sac':
        alg_name = 'sac'
    else:
        if args.robust_feature:
            alg_sub_name = args.alg + '_robust_True'
        else:
            alg_sub_name = args.alg + '_robust_False'
        alg_name = f'{alg_sub_name}_nystrom_{use_nystrom}_rf_num_{args.rf_num}_learn_rf_{args.learn_rf}'
        if use_nystrom:
            alg_name = alg_name + f'_sample_dim_{args.nystrom_sample_dim}'
    exp_name = f'seed_{args.seed}_radius_{args.robust_radius}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}_{args.notes}'

    # setup log
    log_path = f'log/{env_name}/{alg_name}/{exp_name}'
    summary_writer = SummaryWriter(log_path + "/summary_files")

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if args.learn_rf == "False":
        learn_rf = False
    else:
        learn_rf = True

    kwargs = vars(args)
    kwargs.update({
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_space": env.action_space,
        'obs_space_high': np.clip(env.observation_space.high, -3., 3.).tolist(),
        'obs_space_low': np.clip(env.observation_space.low, -3., 3.).tolist(),  # in case of inf observation space
        'obs_space_dim': env.observation_space.shape,
        'dynamics_type': args.env.split('-')[0],
        'dynamics_parameters': {
            'stabilizing_target': [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        },
    })

    kwargs['dynamics_parameters'].update(ENV_CONFIG)

    # Initialize policy
    if args.alg == "sac":
        agent = sac_agent.SACAgent(**kwargs)
    elif args.alg == 'rfsac':
        agent = rfsac_agent.RFSACAgent(**kwargs)

    replay_buffer = buffer.ReplayBuffer(state_dim, action_dim, device=torch.device(args.device))

    # Evaluate untrained policy
    evaluations = [] # util.eval_policy(agent, eval_env)[1]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    timer = util.Timer()

    # keep track of best eval model's state dict
    best_eval_reward = -1e6
    best_actor = None
    best_critic = None
    data_dumped = False

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1
        data_dumped = False
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, explore=True)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, next_state, reward, done)

        prev_state = np.copy(state)
        state = next_state
        episode_reward += reward
        info = {}

        if t >= args.start_timesteps:
            info = agent.train(replay_buffer, batch_size=args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} use_nystrom:{use_nystrom} rf_num:{args.rf_num} seed:{args.seed}")
            # Reset environment
            info.update({'ep_len': episode_timesteps})
            state, done = env.reset(), False
            # prev_state = np.copy(state)
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # dump data

        if (t + 1) % 10 == 0 and t >= args.start_timesteps:
            for key, value in info.items():
                if 'dist' not in key:
                    summary_writer.add_scalar(f'info/{key}', value, t + 1)
                else:
                    for dist_key, dist_val in value.items():
                        summary_writer.add_histogram(dist_key, dist_val, t + 1)
            summary_writer.flush()
            data_dumped = True

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            steps_per_sec = timer.steps_per_sec(t + 1)
            eval_len, eval_ret, _, _ = util.eval_policy(agent, eval_env, eval_episodes=50)
            evaluations.append(eval_ret)

            if t >= args.start_timesteps:
                info.update({'eval_len': eval_len,
                             'eval_ret': eval_ret})
                for key, value in info.items():
                    if 'dist' not in key:
                        summary_writer.add_scalar(f'info/{key}', value, t + 1)
                    else:
                        for dist_key, dist_val in value.items():
                            summary_writer.add_histogram(dist_key, dist_val, t + 1)
                summary_writer.flush()

            print('Step {}. Steps per sec: {:.4g}.'.format(t + 1, steps_per_sec))

            if eval_ret > best_eval_reward:
                best_actor = agent.actor.state_dict()
                best_critic = agent.critic.state_dict()

            best_eval_reward = max(evaluations)

    summary_writer.close()

    print('Total time cost {:.4g}s.'.format(timer.time_cost()))

    # save best actor/best critic
    torch.save(best_actor, log_path + "/actor.pth")
    torch.save(best_critic, log_path + "/critic.pth")

    torch.save(agent.actor.state_dict(), log_path + "/actor_last.pth")
    torch.save(agent.critic.state_dict(), log_path + "/critic_last.pth")

    # save parameters
    with open(os.path.join(log_path, 'train_params.pkl'), 'wb') as fp:
        pkl.dump(kwargs, fp)
