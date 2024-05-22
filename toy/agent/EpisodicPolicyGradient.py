import numpy as np
import random

from . import Agent


class EpisodicPolicyGradientAgent(Agent):
    def __init__(self, env, alpha, H):
        # Environment information.
        self.env            = env
        
        self.num_states     = env.num_states
        self.num_actions    = env.num_actions
        self.states         = env.states
        self.actions        = env.actions
        self. H             = H

        self.reward         = env.reward

        # Learning parameters.
        self.alpha    = alpha

        # Internal state.
        self.pi      = None
    

    # Core functions.
    def reset(self, pi_init):
        assert pi_init.shape == (self.H, self.num_states, self.num_actions)
        self.pi = pi_init
        
        return self.pi
    
    def update(self):
        Q_pi = self.env.robust_Q_SDP(pi=self.pi)

        # Natural policy gradient via soft-max parametrization.
        self.pi = self.pi * np.exp(self.alpha * Q_pi)
        Z = np.sum(self.pi, axis=2)[:,:,np.newaxis]
        self.pi = np.divide(self.pi, Z)
        
        return self.pi, {"Q_pi": Q_pi}

    def select_action(self, h, state):
        return random.choices(self.actions, weights=self.pi[h,int(state),:])[0]
    
    def save(self, path):
        np.save(path, self.pi)
    
    def load(self, path):
        self.pi = np.load(path)