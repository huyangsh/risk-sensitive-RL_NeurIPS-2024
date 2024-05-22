import numpy as np
from math import sin, cos, pi
from . import RMDP

# Define reward_src vectors.
def get_reward_src(env_name):
    if env_name == "Toy-10":
        return np.array([0,0,-1,2,-1,0,0,-10,5,-10,0,0.9,1,0])
    elif env_name == "Toy-100_design":
        return np.array([
            0,-10,5,-10,0,1,3,0,-1,4,-1,0,-1,-1,0,
            0,3,-8,3,0,1,2,0,1,-4,1,0,-1,-2,0,
            0,-12,6,-10,0,4,1,0,-2,3,-1,0,-2,-1,0,
            0,2,-9,4,0,1,2,0,1,-5,2,0,-1,-3,0,
            0,-4,2,-8,0,3,1,0,-2,4,-5,0,-2,-1,0,
            0,3,-6,1,0,1,2,0,1,-6,1,0,-1,-4,0,
            0,-7,3,-7,0,0,2,-6,4,0
        ])
        """ reward_src = np.array([
            0,-10,5,-10,0,1,1,0,-1,2,-1,0,-1,-1,0,
            0,5,-10,5,0,1,1,0,1,-2,1,0,-1,-1,0,
            0,-10,5,-10,0,1,1,0,-1,2,-1,0,-1,-1,0,
            0,5,-10,5,0,1,1,0,1,-2,1,0,-1,-1,0,
            0,-10,5,-10,0,1,1,0,-1,2,-1,0,-1,-1,0,
            0,5,-10,5,0,1,1,0,1,-2,1,0,-1,-1,0,
            0,-10,5,-10,0,0,5,-10,5,0
        ]) """
    elif env_name == "Toy-100_Fourier":
        return np.array([sin(i*pi/100) + cos(2*i*pi/100) + sin(3*i*pi/100) for i in range(100)])
    elif env_name == "Toy-100_zone":
        return np.array( [3*np.sin(i*np.pi/2.05) for i in range(50)] + [np.sin(i*np.pi/50) for i in range(50)] )
    elif env_name == "Toy-100_zone2":
        return np.array( [4*np.sin(i*np.pi/2.05)-1 for i in range(50)] + [np.sin(i*np.pi/50) for i in range(50)] )
    elif env_name == "Toy-1000":
        return np.array([sin(i*pi/1000) + cos(2*i*pi/1000) + sin(3*i*pi/1000) for i in range(1000)])
    else:
        raise NotImplementedError

# Toy environment build functions.
def build_toy_env(reward_src, p_perturb, beta, gamma, thres=1e-5, calc_opt=True):
    num_states  = reward_src.shape[0]
    num_actions = 3    # 0 = left, 1 = stay, 2 = right.
    
    reward = np.zeros(shape=(num_states,num_actions), dtype=np.float64)
    prob = np.zeros(shape=(num_states,num_actions,num_states), dtype=np.float64)
    for s in range(num_states):
        for a in range(num_actions):
            reward[s,a] = reward_src[s]
            
            prob[s,a,(s+a-1)%num_states] = 1 - 2*p_perturb
            prob[s,a,(s+a-2)%num_states] = p_perturb
            prob[s,a,(s+a)%num_states]   = p_perturb

    distr_init = np.ones(shape=(num_states,), dtype=np.float64) / num_states

    return RMDP(num_states, num_actions, distr_init, reward, prob, beta, gamma, thres, calc_opt)