import pickle as pkl
import numpy as np
from envs.env_helper import *
import argparse
import os
from main import ENV_CONFIG
import torch
from agent.rfsac.rfsac_agent import RFVCritic, nystromVCritic
from agent.rfsac import rfsac_agent
from agent.sac import sac_agent
from agent.sac.actor import DiagGaussianActor
from utils.util import eval_policy
import seaborn as sns
import pandas as pd

def eval_robust(log_path, robust='robust'):
    with open(os.path.join(log_path, 'train_params.pkl'), 'rb') as f:        
        kwargs = pkl.load(f)
    env_name = kwargs['env']
    
    eval_config = ENV_CONFIG.copy()
    eval_config.update({'reward_scale': 1.,
                        'eval': True,
                        'reward_exponential': False,
                        'reward_type': 'energy',
                        'noise_scale': kwargs['sigma']})
    dfs = []
    if env_name == "Pendulum-v1":
        m_list = np.linspace(0.2, 1.8, 5)

    elif env_name == 'Quadrotor2D-v2':
        m_list = np.linspace(0.2, 1.8, 5)
    elif env_name == 'Pendubot-v0':
        m_list = np.linspace(0.2, 1.8, 5)
    # elif env_name == 'CartPoleContinuous-v0':
    #     eval_env = env_creator_cartpole(eval_config)
    elif env_name == 'CartPendulum-v0':
        m_list = [0.2, 0.4,0.8,1.0,1.5]
    else:
        raise ValueError('Unknown env name')

    # for g in np.linspace(10, 13, 2):
    for m in m_list:
        eval_config.update(dict(m=m))
        if env_name == "Pendulum-v1":
            eval_env = env_creator_pendulum(eval_config)
        elif env_name == 'Quadrotor2D-v2':
            eval_env = env_creator_quad2d(eval_config)
        elif env_name == 'Pendubot-v0':
            eval_env = env_creator_pendubot(eval_config)
        # elif env_name == 'CartPoleContinuous-v0':
        #     eval_env = env_creator_cartpole(eval_config)
        elif env_name == 'CartPendulum-v0':
            eval_env = env_creator_cartpendulum(eval_config)
        else:
            raise NotImplementedError
        eval_env = Gymnasium2GymWrapper(eval_env)
        kwargs['action_space'] = eval_env.action_space
        kwargs.update({'eval': True})
        if kwargs['alg'] == "sac":
            agent = sac_agent.SACAgent(**kwargs)
        elif kwargs['alg'] == 'rfsac':
            agent = rfsac_agent.RFSACAgent(**kwargs)

        actor = DiagGaussianActor(obs_dim=kwargs['obs_space_dim'][0],
                                  action_dim=kwargs['action_dim'],
                                  hidden_dim=kwargs['hidden_dim'],
                                  hidden_depth=2,
                                  log_std_bounds=[-5., 2.])

        actor.load_state_dict(torch.load(log_path+"/actor.pth"))
        # critic.load_state_dict(torch.load(log_path + "/critic.pth"))
        agent.actor = actor
        agent.device = torch.device("cpu")
        # agent.critic = critic
        print(m)

        _, _, _, ep_rets = eval_policy(agent, eval_env, eval_episodes=50)
        data = {'m':m, 'rets': ep_rets}
        dfs.append(pd.DataFrame.from_dict(data))
        # print(dfs)
    return pd.concat(dfs, ignore_index=True)


if __name__ == '__main__':
    log_paths = {'Quadrotor':{},
                 'noisy_Quadrotor':{},
                 'Pendubot':{},
                 'noisy_Pendubot':{},
                 'CartPoleContinuous':{},
                 'noisy_CartPoleContinuous':{},
                 'CartPendulum': {},
                 'noisy_CartPendulum':{}}
    eval_robust('/home/haitong/PycharmProjects/lvrep-rl-cloned/log/Pendulum-v1_sigma_0.0_rew_scale_1.0/rfsac_nystrom_False_rf_num_512_learn_rf_False/seed_0_2024-05-13-19-24-47_correct_norm_penalty')
    # eval_robust('/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendulum-v1_sigma_0.0_rew_scale_1.0/rfsac_nystrom_False_rf_num_512_learn_rf_False/seed_0_2024-02-01-10-45-39')