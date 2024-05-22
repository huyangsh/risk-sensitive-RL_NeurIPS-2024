from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium
import numpy as np
from gymnasium.core import ObsType, ActType


class Quadrotor2D(gymnasium.Env):
    # dynamics params
    m = 0.027
    g = 10.0
    Iyy = 1.4e-5
    dt = 0.008
    stabilizing_target = np.array([0., 0., 0.5, 0., 0., 0.])

    def __init__(self, sin_input = True, eval = False, noisy=False, noise_scale=0.0, **kwargs):
        super(Quadrotor2D, self).__init__()
        self.eval = eval
        self.sin_input = sin_input
        m = kwargs.get('m', 1.0)
        self.m = 0.027 * m
        self.action_space = gymnasium.spaces.Box(-1 * np.ones([2,]), np.ones([2,]))
        obs_low = np.array([-2., -np.inf, -0.05, -np.inf, -1.4835298, -np.inf ], dtype=float)
        obs_high = np.array([2., np.inf, 3., np.inf, 1.4835298, np.inf ], dtype=float)
        self.obs_high = obs_high
        self.observation_space = gymnasium.spaces.Box(obs_low, obs_high)
        self.state = np.zeros([6,])
        self.action = None
        self.action_scale = 1.
        if noisy or noise_scale > 0.:
            self.sigma = noise_scale
        else:
            self.sigma = None

    def preprocess_action(self, action):
        action = 0.075 * action + 0.075 # map from -1, 1 to 0.0 - 0.15
        # print(action)
        return action

    def get_normalized_action(self, force):
        action = (force - 0.075) / 0.075
        return action

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self.state = np.array([0., 0., 0.5, 0., 0., 0.])
        init_rand = np.random.uniform(size=(6,))
        init_rand = np.multiply(np.array([0.1, 0.01, 0.1, 0.01, 0.1, 0.01]), init_rand)
        self.state = self.state + init_rand
        return self.state, {}

    def quadrotor_f_star_6d(self, states, action):
        m = self.m
        Iyy = self.Iyy
        dt = self.dt
        g = self.g
        def dot_states(states):
            dot_states = np.hstack([states[1],
                                    1 / m * np.multiply(np.sum(action), np.sin(states[4])),
                                    states[3],
                                    1 / m * np.multiply(np.sum(action), np.cos(states[4])) - g,
                                    states[5],
                                    1 / 2 / Iyy * (action[1] - action[0]) * 0.025
                                    ])
            return dot_states

        k1 = dot_states(states)
        k2 = dot_states(states + dt / 2 * k1)
        k3 = dot_states(states + dt / 2 * k2)
        k4 = dot_states(states + dt * k3)

        return states + dt / 6 * (k1 + k2 + k3 + k4)


    def get_reward(self, state, action):
        state_error = state - self.stabilizing_target
        reward = - np.sum(np.multiply(np.array([1., 0., 1., 0., 0., 0.]),
                                      state_error ** 2))#  + np.sum(0.1 * action ** 2)
        return reward

    def get_done(self):
        done = np.any(np.abs(self.state) >= self.obs_high)
        return done, np.abs(self.state) >= self.obs_high

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = self.preprocess_action(action)
        old_state = self.state
        self.state = self.quadrotor_f_star_6d(self.state, action)
        done, done_where = self.get_done()
        if self.eval:
            done = False
        if self.sigma is not None:
            noise = np.random.normal(scale=self.sigma * self.dt, size=[3,])
            obs = self.state
            obs[0] = obs[0] + noise[0]
            obs[2] = obs[2] + noise[1]
            obs[4] = obs[4] + noise[2]
        else:
            obs = self.state
        reward = self.get_reward(old_state, action) # -100. if done else
        return obs, reward, done, False, {
            'done_where': done_where
        }

    def get_obs(self):
        return self.state

    def get_energy_based_control(self, Tstable = 0.0, p_t = 0.1, ke = 1., kx = 1.):
        m = self.m
        Iyy = self.Iyy
        dt = self.dt
        g = self.g
        x, xdot, z, zdot, th, thdot = self.state
        total_T = Tstable + p_t * (z - 0.5)
        F = kx * xdot + (ke * m * g + kx) * zdot
        T_diff = - np.sqrt(2) * (thdot ** 2 + F) / (ke * thdot)
        T2 = (total_T + T_diff) / 2
        T1 = (total_T - T_diff) / 2
        action = np.clip(np.array([T1, T2]), 0., 0.15)
        return self.get_normalized_action(action)


if __name__ == '__main__':
    from gymnasium.envs.registration import register
    register('Quadrotor2D-v2', Quadrotor2D, max_episode_steps=10)
    env = gymnasium.make('Quadrotor2D-v2', noisy=True, noise_scale=1.)
    print(env.observation_space)
    print(env.action_space)
    rets = []
    for i in range(50):
        ret = 0.
        env.reset()
        for i in range(200):
            _, rew, _, _, _ = env.step(env.get_energy_based_control())
            ret += rew
        rets.append(ret)

    print(np.mean(rets), np.std(rets))




