import numpy as np
import random
import pickle as pkl

from . import Agent


class RFZI_Tabular(Agent):
    def __init__(self, env):
        # Environment information.
        self.env            = env

        self.num_states     = env.num_states
        self.num_actions    = env.num_actions
        self.states         = env.states
        self.actions        = env.actions

        self.reward         = env.reward
        self.beta           = env.beta
        self.gamma          = env.gamma
        
        # Internal state.
        self.Z  = None
        self.pi = None


    # Core functions.
    def reset(self, Z_init):
        assert Z_init.shape == (self.num_states, self.num_actions)
        self.Z = Z_init
        self.pi = None

    def update(self, dataset, verbose=True):
        num_data = len(dataset)

        # Calculate the maximal risk measure (best response) for each state.
        next_rewards = np.exp( (self.reward - np.log(self.Z)/self.beta).max(axis=1) * (-self.beta*self.gamma) )
        
        # Calculate Z_next.
        Z_next = np.zeros_like(self.Z)
        cnt = np.zeros_like(self.Z)
        for s,a,r,s_,_ in dataset:
            s, a, s_ = int(s), int(a), int(s_)
            Z_next[s,a] += next_rewards[s_]
            cnt[s,a] += 1
        Z_next[np.where(cnt > 0)] /= cnt[np.where(cnt > 0)]
        Z_next[np.where(cnt == 0)] = self.Z[np.where(cnt == 0)]

        # Logging.
        info = {}
        if verbose:
            # Training loss.
            loss = 0
            for s,a,r,s_,_ in dataset:
                s, a, s_ = int(s), int(a), int(s_)
                loss += (Z_next[s,a] - next_rewards[s_]) ** 2
            loss /= num_data

            info = {
                "loss": loss,
                "diff": self.Z - Z_next
            }

        # Update.
        self.Z = Z_next
        return self.Z, info

    def select_action(self, state):
        if self.pi is None: self.get_policy()
        return random.choices(self.actions, weights=self.pi[state,:])[0]
    
    def save(self, path):
        with open(path, "wb") as f:
            pkl.dump({"pi": self.pi, "Z":self.Z}, f)
    
    def load(self, path):
        with open(path, "rb") as f:
            data = pkl.load(f)
        self.pi = data["pi"]
        self.Z = data["Z"]


    # Utility: generate policy matrix.
    def get_policy(self):
        a_max = (self.reward - self.Z/self.beta).argmax(axis=1)
        pi = np.zeros(shape=(self.num_states,self.num_actions), dtype=np.float64)
        pi[np.arange(self.num_states), a_max] = 1

        self.pi = pi
        return pi