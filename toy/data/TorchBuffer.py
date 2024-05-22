import numpy as np
import torch
import pickle as pkl

class TorchBuffer:
    def __init__(self, device, dim_state, dim_action, max_size, off_ratio=0.99):
        self.device = device

        self.dim_state  = dim_state
        self.dim_action = dim_action
        self.max_size   = max_size

        # Initialization.
        self.state      = torch.zeros((max_size, self.dim_state), device=device)
        self.action     = torch.zeros((max_size, self.dim_action), device=device)
        self.reward     = torch.zeros((max_size, 1), device=device)
        self.next_state = torch.zeros((max_size, self.dim_state), device=device)
        self.not_done   = torch.zeros((max_size, 1), device=device)

        self.size       = 0
        self.off_ratio  = off_ratio
        self.ptr        = 0
        self.min_ptr    = int(self.max_size*self.off_ratio)


    # Core functions: data collection.
    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr]        = torch.FloatTensor(state).to(self.device)
        self.action[self.ptr]       = torch.FloatTensor(action).to(self.device)
        self.reward[self.ptr]       = torch.FloatTensor([reward]).to(self.device)
        self.next_state[self.ptr]   = torch.FloatTensor(next_state).to(self.device)
        self.not_done[self.ptr]     = torch.FloatTensor([1. - done]).to(self.device)

        self.ptr += 1
        if self.ptr >= self.max_size: self.ptr = self.min_ptr
        self.size = min(self.size + 1, self.max_size)


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
    def save(self, filename):  # Warning: wrong if load_size < max_size and the blank is not filled.
        with open(filename, "wb") as f:
            pkl.dump({
                "size":         self.size,
                "state":        self.state,
                "action":       self.action,
                "reward":       self.reward, 
                "next_state":   self.next_state,
                "not_done":     self.not_done
            }, f)

    def load(self, filename, shuffle=False):
        with open(filename, "rb") as f:
            data = pkl.load(f)
        
        if shuffle:
            idx = torch.randperm(n=data["size"])
            data["state"]       = data["state"][idx]
            data["action"]      = data["action"][idx]
            data["reward"]      = data["reward"][idx]
            data["next_state"]  = data["next_state"][idx]
            data["not_done"]    = data["not_done"][idx]
        
        if data["size"] < self.max_size:
            self.size                   = data["size"]
            self.state[:self.size]      = data["state"].to(self.device)
            self.action[:self.size]     = data["action"].to(self.device)
            self.reward[:self.size]     = data["reward"].to(self.device)
            self.next_state[:self.size] = data["next_state"].to(self.device)
            self.not_done[:self.size]   = data["not_done"].to(self.device)

            self.ptr                    = self.size
            assert self.ptr >= self.min_ptr
        else:
            self.size       = self.max_size
            self.state      = data["state"][:self.size].to(self.device)
            self.action     = data["action"][:self.size].to(self.device)
            self.reward     = data["reward"][:self.size].to(self.device)
            self.next_state = data["next_state"][:self.size].to(self.device)
            self.not_done   = data["not_done"][:self.size].to(self.device)
            
            self.ptr        = self.min_ptr