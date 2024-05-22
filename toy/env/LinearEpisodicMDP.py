import numpy as np
import scipy as sp
import cvxpy as cp
from scipy.optimize import minimize
import random
from functools import partial
from copy import deepcopy
from math import exp, log

from . import Env


class LinearEpisodicMDP(Env):
    def __init__(self, distr_init, phi, theta, mu, H, eps_phi=0, eps_theta=0, eps_mu=0, thres=1e-5):
        self.num_states, self.num_actions, self.dim_feature = phi.shape
        assert distr_init.shape == (self.num_states,)
        assert theta.shape == (self.dim_feature,)
        assert mu.shape == (self.num_states, self.dim_feature)
        
        self.states      = np.arange(self.num_states)
        self.actions     = np.arange(self.num_actions)
        self.dim_state   = 1
        self.dim_action  = 1

        self.distr_init  = distr_init

        self.phi    = phi
        self.theta  = theta
        self.mu     = mu

        self.H      = H

        self.reward = phi @ theta
        self.prob   = np.squeeze(mu @ phi[:,:,:,np.newaxis])

        self.thres  = thres

        # Perturbation radii.
        self.eps_phi     = eps_phi
        self.eps_theta   = eps_theta
        self.eps_mu      = eps_mu


    # Environment functions (compatible with OpenAI gym).
    def reset(self):
        self.state = random.choices(self.states, weights=self.distr_init)[0]
        return np.array([self.state], dtype=np.float32)
    
    def step(self, action):
        reward = self.reward[self.state, action]
        self.state = random.choices(self.states, weights=self.prob[self.state,action,:])[0]
        return np.array([self.state], dtype=np.float32), reward, False, None    # Compatible with the OpenAI gym interface: done = False (non-episodic).

    def episodic_eval(self, policy, verbose=False):
        state, done = self.reset(), False
        reward_tot = 0
        if verbose: trajectory = []
        for h in range(self.H):
            action = policy(h, state)
            next_state, reward, done, _ = self.step(action)

            reward_tot += reward
            if verbose: trajectory.append([state, action, reward, next_state, done])

            state = next_state
        
        if verbose:
            return reward_tot, trajectory
        else:
            return reward_tot

    # Utility: standard policy evaluation via DP.
    def DP_opt(self):
        V = np.zeros(shape=(self.H+1, self.num_states), dtype=np.float32)

        for h in range(self.H-1, -1, -1):
            for s in self.states:
                reward_max = None
                for a in self.actions:
                    V_pi_cum = 0
                    for s_ in self.states:
                        V_pi_cum += self.prob[s,a,s_] * V[h+1,s_]

                    if reward_max is None:
                        reward_max = self.reward[s,a] + V_pi_cum
                    else:
                        reward_max = max(reward_max, self.reward[s,a] + V_pi_cum)
                
                V[h,s] = reward_max
        
        return V[:self.H, :]
    
    def DP_pi(self, pi):
        assert pi.shape == (self.H, self.num_states, self.num_actions)
        V = np.zeros(shape=(self.H+1, self.num_states), dtype=np.float32)

        for h in range(self.H-1, -1, -1):
            for s in self.states:
                for a in self.actions:
                    V_pi_cum = 0
                    for s_ in self.states:
                        V_pi_cum += self.prob[s,a,s_] * V[h+1,s_]

                    V[h,s] += pi[h,s,a] * (self.reward[s,a] + V_pi_cum)
        
        return V[:self.H, :]
    
    def V_to_Q(self, V):
        assert V.shape == (self.H, self.num_states)
        Q = np.zeros(shape=(self.H, self.num_states, self.num_actions), dtype=np.float32)
        for h in range(self.H-1, -1, -1):
            for s in self.states:
                for a in self.actions:
                    V_pi_cum = 0
                    if h < self.H-1:
                        for s_ in self.states:
                            V_pi_cum += self.prob[s,a,s_] * V[h+1,s_]

                    Q[h,s,a] = self.reward[s,a] + V_pi_cum
        
        return Q
    
    def Q_to_V(self, Q, pi):
        assert Q.shape == (self.H, self.num_states, self.num_actions)
        assert pi.shape == (self.H, self.num_states, self.num_actions)
        return np.sum(Q*pi, axis=2)
    
    def Q_to_pi(self, Q):
        assert Q.shape == (self.H, self.num_states, self.num_actions)
        indices = np.argmax(Q, axis=2)[:,:,np.newaxis]
        pi = np.zeros(shape=(self.H, self.num_states, self.num_actions), dtype=np.float32)
        np.put_along_axis(pi, indices=indices, values=1, axis=2)
        return pi
    
    def get_opt_pi(self):
        V_opt  = self.DP_opt()
        Q_opt  = self.V_to_Q(V_opt)
        pi_opt = self.Q_to_pi(Q_opt)
        return pi_opt
    

    # Utility: robust policy evaluation.
    def robust_Q_SDP(self, pi):
        assert pi.shape == (self.H, self.num_states, self.num_actions)
        rho_pi = self.visit_freq(pi)
            
        V = np.zeros(shape=(self.H+1,self.num_states), dtype=np.float32)
        Q = np.zeros(shape=(self.H,self.num_states,self.num_actions), dtype=np.float32)
        for h in range(self.H-1, -1, -1):
            const_xi = self.theta + self.mu.T @ V[h+1,:]
            const_eta = np.sum((rho_pi[h,:][:,np.newaxis] * pi[h,:,:])[:,:,np.newaxis] * self.phi, axis=(0,1))
            # V_clipped = np.minimum(abs(V), np.ones(shape=(self.num_states,)) * (self.H-h+1))
            # eps_xi = self.eps_theta + (self.H-h+1) * self.eps_mu
            eps_xi = self.eps_mu

            # Optimization step: reduction to SDP.
            n = self.dim_feature
            A = np.vstack([
                np.hstack([np.zeros(shape=(n,n)), np.eye(N=n)]),
                np.hstack([np.eye(N=n), np.zeros(shape=(n,n))])
            ]) * 0.5
            beta = np.hstack([const_eta.flatten(), const_xi.flatten()])[:, np.newaxis] * 0.5

            C = np.vstack([
                np.hstack([A, beta]),
                np.hstack([beta.T, [[0]]])
            ])

            Cx = sp.linalg.block_diag(np.eye(N=n), np.zeros(shape=(n,n)), [-eps_xi**2])
            Cy = sp.linalg.block_diag(np.zeros(shape=(n,n)), np.eye(N=n), [-self.eps_phi**2])
            C0 = sp.linalg.block_diag(np.zeros(shape=(2*n,2*n)), [1])
            # Create a symmetric matrix variable.
            X = cp.Variable((2*n+1,2*n+1), symmetric=True)
            # Create constraints.
            constraints = [
                X >> 0,  # The operator >> denotes matrix inequality.
                cp.trace(Cx @ X) <= 0,
                cp.trace(Cy @ X) <= 0,
                cp.trace(C0 @ X) == 1
            ]  
            problem = cp.Problem(
                cp.Minimize(cp.trace(C @ X)),
                constraints
            )
            problem.solve()

            z = X.value[-1, :-1]
            xi_h = z[:n]
            eta_h = z[n:]

            
            # Update omega and V.
            omega_h  = const_xi + xi_h
            phi_h    = self.phi + eta_h
            Q[h,:,:] = phi_h @ omega_h
            V[h,:]   = np.sum(Q[h,:,:] * pi[h,:,:], axis=1)

        return Q


    # Utility: calculate state-visit frequency.
    def _transit(self, distr, prob, pi):
        distr_new = np.zeros(shape=(self.num_states,), dtype=np.float32)
        for s in self.states:
            for a in self.actions:
                for s_ in self.states:
                    distr_new[s_] += distr[s] * pi[s,a] * prob[s,a,s_]
        
        return distr_new

    def visit_freq(self, pi):
        assert pi.shape == (self.H, self.num_states, self.num_actions)
        distr_cur = deepcopy(self.distr_init)

        d_pi = np.zeros(shape=(self.H, self.num_states), dtype=np.float32)
        for h in range(self.H):
            d_pi[h, :] = distr_cur
            distr_cur = self._transit(distr_cur, self.prob, pi[h,:,:])
        
        return d_pi