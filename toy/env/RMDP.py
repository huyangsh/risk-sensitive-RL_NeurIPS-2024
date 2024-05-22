import numpy as np
import random
from copy import deepcopy
from math import exp, log

from . import Env


class RMDP(Env):
    def __init__(self, num_states, num_actions, distr_init, reward, prob, beta, gamma, thres=1e-5, calc_opt=True):
        assert distr_init.shape == (num_states,)
        assert reward.shape == (num_states, num_actions)
        assert prob.shape == (num_states, num_actions, num_states)
        assert gamma <= 1

        self.num_states  = num_states
        self.num_actions = num_actions
        self.states      = np.arange(self.num_states)
        self.actions     = np.arange(self.num_actions)
        self.distr_init  = distr_init

        self.dim_state   = 1
        self.dim_action  = 1

        self.reward = reward
        self.prob   = prob

        self.beta   = beta
        self.gamma  = gamma
        self.coeff  = gamma / beta

        self.thres  = thres

        # Utility: precalculated optimal risk measure.
        if calc_opt:
            self.V_opt     = self._DP_opt(thres=self.thres)
            self.V_opt_avg = (self.V_opt*self.distr_init).sum()


    # Environment functions (compatible with OpenAI gym).
    def reset(self):
        self.state = random.choices(self.states, weights=self.distr_init)[0]
        return np.array([self.state], dtype=np.float32)
    
    def step(self, action):
        reward = self.reward[self.state, action]
        self.state = random.choices(self.states, weights=self.prob[self.state,action,:])[0]
        return np.array([self.state], dtype=np.float32), reward, False, None    # Compatible with the OpenAI gym interface: done = False (non-episodic).


    # Utility: Bellman updates (using DP). 
    def _DP_opt(self, thres):
        V = np.zeros(shape=(self.num_states,), dtype=np.float64)

        diff = thres + 1
        while diff > thres:
            V_prev = V
            V = np.zeros(shape=(self.num_states,), dtype=np.float64)

            for s in self.states:
                reward_max = None
                for a in self.actions:
                    V_pi_cum = 0
                    for s_ in self.states:
                        V_pi_cum += self.prob[s,a,s_] * exp(-self.beta * V_prev[s_])

                    if reward_max is None:
                        reward_max = self.reward[s,a] - self.coeff * log(V_pi_cum)
                    else:
                        reward_max = max(reward_max, self.reward[s,a] - self.coeff * log(V_pi_cum))
                
                V[s] = reward_max
            
            diff = np.linalg.norm(V - V_prev)
        
        return V

    def DP_pi(self, pi, thres):
        assert pi.shape == (self.num_states, self.num_actions)
        V = np.zeros(shape=(self.num_states,), dtype=np.float64)

        diff = thres + 1
        while diff > thres:
            V_prev = V
            V = np.zeros(shape=(self.num_states,), dtype=np.float64)

            for s in self.states:
                for a in self.actions:
                    V_pi_cum = 0
                    for s_ in self.states:
                        V_pi_cum += self.prob[s,a,s_] * exp(-self.beta * V_prev[s_])

                    V[s] += pi[s,a] * (self.reward[s,a] - self.coeff * log(V_pi_cum))
            
            diff = np.linalg.norm(V - V_prev)
        
        return V
    
    def V_to_Q(self, V):
        assert V.shape == (self.num_states,)
        Q = np.zeros(shape=(self.num_states, self.num_actions), dtype=np.float64)
        for s in self.states:
            for a in self.actions:
                V_pi_cum = 0
                for s_ in self.states:
                    V_pi_cum += self.prob[s,a,s_] * exp(-self.beta * V[s_])

                Q[s,a] = self.reward[s,a] - self.coeff * log(V_pi_cum)
        
        return Q
    

    # Utility: calculate state-visit frequency.
    def _prob_hat(self, pi, V_pi):
        if V_pi is None: V_pi = self.DP_pi(pi, thres=self.thres)
        V_pi = V_pi[np.newaxis, np.newaxis, :]
        
        prob_hat = self.prob * np.exp(-self.beta*V_pi)
        prob_hat /= prob_hat.sum(axis=2, keepdims=True)
        return prob_hat
    
    def _transit(self, distr, prob, pi):
        distr_new = np.zeros(shape=(self.num_states,), dtype=np.float64)
        for s in self.states:
            for a in self.actions:
                for s_ in self.states:
                    distr_new[s_] += distr[s] * pi[s,a] * prob[s,a,s_]
        
        return distr_new

    def visit_freq(self, pi, T, V_pi=None):
        assert pi.shape == (self.num_states, self.num_actions)
        prob_hat = self._prob_hat(pi, V_pi)

        distr_cur = deepcopy(self.distr_init)
        g_t = 1
        d_pi = distr_cur
        for t in range(T):
            g_t *= self.gamma
            distr_cur = self._transit(distr_cur, prob_hat, pi)
            d_pi += g_t * distr_cur
        
        d_pi *= 1-self.gamma
        return d_pi