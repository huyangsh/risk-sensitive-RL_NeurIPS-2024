import numpy as np
import torch
import pickle as pkl

class TorchDataset:
    def __init__(self, device):
        self.device = device

        # Internal state.
        self.size = 0


    # Core functions: data collection.
    def start(self, dim_state, dim_action, max_size):
        # Parameters.
        self.dim_state  = dim_state
        self.dim_action = dim_action
        self.max_size   = max_size

        # Initialization.
        self.state      = np.zeros((max_size, self.dim_state))
        self.action     = np.zeros((max_size, self.dim_action))
        self.reward     = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, self.dim_state))
        self.not_done   = np.zeros((max_size, 1))

        self.ptr        = 0
        self.size       = 0

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr]        = state
        self.action[self.ptr]       = action
        self.reward[self.ptr]       = reward
        self.next_state[self.ptr]   = next_state
        self.not_done[self.ptr]     = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def finish(self):
        self.state      = torch.FloatTensor(self.state).to(self.device)
        self.action     = torch.FloatTensor(self.action).to(self.device)
        self.reward     = torch.FloatTensor(self.reward).to(self.device)
        self.next_state = torch.FloatTensor(self.next_state).to(self.device)
        self.not_done   = torch.FloatTensor(self.not_done).to(self.device)


    # Core function: data sampling.
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.not_done[ind]
        )


    # Core functions: loading and saving.
    def save(self, filename):
        with open(filename, "wb") as f:
            pkl.dump({
                "size":         self.size,
                "state":        self.state,
                "action":       self.action,
                "reward":       self.reward, 
                "next_state":   self.next_state,
                "not_done":     self.not_done
            }, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            data = pkl.load(f)
        
        self.size       = data["size"]
        self.state      = data["state"].to(self.device)
        self.action     = data["action"].to(self.device)
        self.reward     = data["reward"].to(self.device)
        self.next_state = data["next_state"].to(self.device)
        self.not_done   = data["not_done"].to(self.device)