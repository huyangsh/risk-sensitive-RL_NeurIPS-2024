from abc import abstractmethod

class Env:
    @abstractmethod
    def reset(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    def eval(self, policy, T_eval, verbose=False):
        state, done = self.reset(), False
        reward_tot, g_t, t = 0, 1, 0
        if verbose: trajectory = []
        while not done:
            if t >= T_eval: break
            t += 1

            action = policy(state)
            next_state, reward, done, _ = self.step(action)

            reward_tot += g_t * reward
            g_t *= self.gamma
            if verbose: trajectory.append([state, action, reward, next_state, done])

            state = next_state
        
        if verbose:
            return reward_tot, trajectory
        else:
            return reward_tot
    