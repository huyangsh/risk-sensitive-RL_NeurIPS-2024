"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
NEW: Adapted from OpenAI Gym.
"""

import math
import numpy as np
import warnings

from . import Env


class CartPole(Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    def __init__(self, sigma=0.0, T_max=0):
        self.gravity         = 9.8
        self.masscart        = 1.0
        self.masspole        = 0.1
        self.total_mass      = self.masspole + self.masscart
        self.length          = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag       = 10.0
        self.tau             = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians
        # so failing observation is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )


        # Environment parameters.
        self.dim_state = 4
        self.dim_action = 1
        self.num_actions = 2
        self.actions = [np.array([0]), np.array([1])]

        self.gamma = 1
        self.sigma = sigma    # magnitude of Gaussian noise

        # Internal states.
        self.state = None
        self.steps_beyond_done = None
        
        self.t     = 0
        self.T_max = T_max


    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert action in self.actions, err_msg

        x, x_dot, theta, theta_dot = self.state
        action = action[0]
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass) + self.sigma * np.random.normal(0,1)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass + self.sigma * np.random.normal(0,1)

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        # upper bound on episode length
        self.t += 1
        if self.T_max > 0 and self.t >= self.T_max:
            done = True

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                warnings.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}
        
    def reset(self, seed=None, length=0.5, gravity=9.8, force_mag=10.0, 
              init_angle_mag=0.05, init_vel_mag=0.05):
        # set seed if any
        if seed is not None:
            np.random.seed(seed)
        
        # upper bound on episode length
        self.t = 0

        # perturbing the environment
        self.gravity         = gravity
        self.force_mag       = force_mag
        self.length          = length
        self.polemass_length = self.masspole * self.length
        init_angle           = np.random.uniform(-init_angle_mag, init_angle_mag)
        init_vel             = np.random.uniform(-init_vel_mag, init_vel_mag)
        # build state
        # self.state[2] is initial angle
        # self.state[3] is initial velocity

        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,)) # changed from 4 to 2
        self.state[2] = init_angle
        self.state[3] = init_vel
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def reward(self, state, action):
        x, x_dot, theta, theta_dot = tuple(state)
        action = action[0]
        
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        # Warning: cannot identify whether the pole has just fallen or not.
        if not done:
            reward = 1.0
        else:
            reward = 0.0
        
        return reward