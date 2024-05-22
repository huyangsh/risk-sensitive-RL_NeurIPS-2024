"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c 
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from noisy_pend import angle_normalize

import envs





class CartPendulumEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ## Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('CartPole-v1')
    ```

    On reset, the `options` parameter allows the user to change the bounds used to determine
    the new random state.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }


    def __init__(self, render_mode: Optional[str] = None, task = 'swingup', noise_scale= 0., eval=False, m=1.0):
        """
        task : swingup or balance
        """
        self.g = 10
        self.M = 0.5 * m
        self.m = 0.2 * m
        self.b = 0.1
        self.I = 0.006
        self.l = 0.3
        self.total_mass = self.m + self.M
        # self.length = 0.5  # actually half the pole's length
        # self.polemass_length = self.m * self.length
        self.force_mag = 0.4
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        self.noise_scale = noise_scale

        self.eval = eval

        # Angle at which to fail the episode
        self.task = task
        if task == 'balance':
            self.theta_threshold_radians = 12 * 2 * math.pi / 360
        elif task == 'swingup':
            self.theta_threshold_radians = np.finfo(np.float32).max
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(-1., 1., dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def angle_normalize(self, theta):
        return np.remainder(theta, 2 * np.pi)

    def step(self, action):
        if isinstance(action, list) or isinstance(action, np.ndarray):
            action = action[0]
        # err_msg = f"{action!r} ({type(action)}) invalid"
        # assert self.action_space.contains(action), err_msg
        # assert self.state is not None, "Call reset before using step method."
        # x, x_dot, theta, theta_dot = self.state
        # # force = self.force_mag if action == 1 else -self.force_mag # discrete action space
        # force = float(self.force_mag * action)
        # costheta = math.cos(theta)
        # sintheta = math.sin(theta)
        #
        # # For the interested reader:
        # # https://coneural.org/florian/papers/05_cart_pole.pdf
        # temp = (
        #     force + self.polemass_length * theta_dot**2 * sintheta
        # ) / self.total_mass
        # thetaacc = (self.gravity * sintheta - costheta * temp) / (
        #     self.length * (4.0 / 3.0 - self.m * costheta ** 2 / self.total_mass)
        # )
        # xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        force = self.force_mag * action

        x, x_dot, theta, theta_dot = self.state
        g, M, m, b, I, l = self.g, self.M, self.m, self.b, self.I, self.l
        # dthdot = (-g / l * torch.sin(th + math.pi) - mu/(m*l**2) + 1/ (m * l ** 2) * ctrl)
        # The system can be written as
        # Az = b
        # for z = (dxdot, dthdot) and A, b given by physics laws (A is symmetric)
        # The following code inverts A to get the expression of z
        a11, a22 = M + m, I + m * l ** 2
        a12 = m * l * np.cos(theta)
        detA = I * (M + m) + m * l ** 2 * M + m ** 2 * l ** 2 * np.sin(theta) ** 2
        b1 = m * l * theta_dot ** 2 * np.sin(theta) - b * x_dot + force
        b2 = -m * g * l * np.sin(theta)
        xacc = (a22 * b1 - a12 * b2) / detA
        thetaacc = (-a12 * b1 + a11 * b2) / detA
        # dx = xdot.unsqueeze(-1)
        # dth = thdot.unsqueeze(-1)
        # dt_state = torch.stack((dx, dth, dxdot, dthdot)).view(-1)

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc + self.noise_scale * self.np_random.normal(scale=0.02)
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc + self.noise_scale * self.np_random.normal(scale=0.05)
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            # reward = 1.0
            if self.task == 'swingup':
                if not self.eval:
                    reward = -float(self.angle_normalize(theta) - np.pi) ** 2
                else:
                    reward = -float(self.angle_normalize(theta) - np.pi) ** 2
            # elif self.task == 'balance':
            #     raise NotImplementedError
            else:
                raise NotImplementedError
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = -1000.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = -1000.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        if self.task == 'swingup':
            self.state[2] = self.np_random.uniform(low=0., high=np.pi) # if swingup, we all start from bottom
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-(x[2] + np.pi))
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def energy_based_controller(self, ke=5., gamma = 1.):
        potential_energy = self.m * self.gravity * 0.5 * self.length * (np.cos(self.state[2]) - 1.)
        kinematics_energy = 0.5 * (self.m * self.state[1] ** 2 + self.m * self.length ** 2 * self.state[3] ** 2 / 3)
        spring_energy = ke / 2 * self.state[0] ** 2 # following the https://ieeexplore.ieee.org/document/6630919
        action = - 0.1 * gamma * (potential_energy + kinematics_energy + spring_energy) * self.state[1] - ke * self.state[0]
        return np.clip(action, self.action_space.low, self.action_space.high)


def cart_pole():
    import time
    env = CartPendulumEnv(render_mode='human')
    print(env.reset())
    env.render()
    import time
    for i in range(100):
        state, _, _, _, _ = env.step(env.action_space.sample())
        print(state[2])
        time.sleep(0.1)
        env.render()

def energy_based_swingup(noise_scale = 1.0):
    env = CartPendulumEnv(noise_scale=noise_scale)
    ep_rets = []
    for i in range(50):
        env.reset(seed=i)
        ep_ret = 0.
        for i in range(300):
            action = env.energy_based_controller()
            _, reward, done, terminated, _ = env.step(action)
            ep_ret += reward
            print(action, done, terminated)
        ep_rets.append(ep_ret)
    print(np.mean(ep_rets), np.std(ep_rets))


if __name__ == '__main__':
    cart_pole()
