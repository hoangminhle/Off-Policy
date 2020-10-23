"""
Cart pole swing-up: Original version from:
https://github.com/zuoxingdong/DeepPILCO/blob/master/cartpole_swingup.py

The original python version is supposed to mirror PILCO V0.9, which is 
a package in Matlab:
http://mlg.eng.cam.ac.uk/pilco/

An adapted version of the cartpole swingup was also used in
https://github.com/hardmaru/estool/blob/master/custom_envs/cartpole_swingup.py

In this version, we will make some modification to allow:
- Discretizing action space
- Choosing between Balancing task and Swingup task
- Choosing the time dt between state updates, smaller means harder problem
- Specifying the horizon and reward structure (default will be smooth reward)
- Integrate into Dilbert style (i.e., reading in the spec file to initialize the environment)
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from rl_nexus.utils.metric import Metric

class CartPoleV2_Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, spec_tree):
        self.gravity = 9.8 # default gravity
        self.masscart = 1.0 # cart mass
        self.masspole = 0.1 # pendulum mass
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02 # seconds between state update
        #* to keep things simple, we will ignore friction coefficient here
        # self.b = 0.1 # friction coefficient
        self.kinematics_integrator = 'euler'

        self.max_ep_len = spec_tree['max_ep_len']

        # Angle at which to fail the episode
        #* depending on the type of task (we may not need the theta threshold)
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self.task = spec_tree['task'] # balance or swingup
        self.num_actions = spec_tree['num_actions']

        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.action_numerical_map = np.linspace(-1.0, 1.0, self.num_actions)

        seed = spec_tree['seed']
        self.seed(seed)
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.fixed_length_episode = spec_tree['fixed_length_episode']
        self.smooth_reward = spec_tree['smooth_reward']
        self.stochastic_dynamics = spec_tree['stochastic_dynamics']

        self.episode_reward = None

        self.reward_metric = Metric(
            short_name='rews',
            long_name='trajectory reward',
            formatting_string='{:5.1f}',
            higher_is_better=True)
        self.time_step_metric = Metric(
            short_name='steps',
            long_name = 'total number of steps',
            formatting_string='{:5.1f}',
            higher_is_better=True)
        self.metrics = [self.reward_metric, self.time_step_metric]
        # get the name and make sure the environment spec is properly specified
        self.name = spec_tree['name']
        if self.name == 'CartPoleV2_Balance_Sparse':
            assert self.task == 'balance', 'wrong task specification for balancing task'
            assert self.smooth_reward == False, 'Rewards do not correspond to sparse setting'
            # assert self.stochastic_dynamics == False, 'Sparse setting should correspond to deterministic dynamics'
        elif self.name == 'CartPoleV2_Balance_Smooth':
            assert self.task == 'balance', 'wrong task specification for balancing task'
            assert self.smooth_reward == True, 'Rewards do not correspond to the smooth setting'
            # assert self.stochastic_dynamics == True, 'Smooth setting should correspond to stochastic dynamics'
        elif self.name == 'CartPoleV2_Swingup_Smooth':
            assert self.task == 'swingup', 'wrong task specification for swingup task'
            assert self.smooth_reward == True, 'Rewards do not correspond to the smooth setting'
        elif self.name == 'CartPoleV2_Swingup_Sparse':
            assert self.task == 'swingup', 'wrong task specification for swingup task'
            assert self.smooth_reward == False, 'Rewards do not correspond to sparse setting'
        else:
            raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.squeeze(action)

        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        #* map action to force
        force = self.action_numerical_map[action] * self.force_mag

        state = self.state
        x, x_dot, theta, theta_dot = state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        
        if self.stochastic_dynamics:
            x += self.np_random.normal(0.0, 0.001, size=x.shape)
            x_dot += self.np_random.normal(0.0, 0.001, size=x_dot.shape)
            theta += self.np_random.normal(0.0, 0.001, size=theta.shape)
            theta_dot += self.np_random.normal(0.0, 0.001, size=theta_dot.shape)

        self.state = (x, x_dot, theta, theta_dot)

        if self.task == 'balance':
            done = bool(
                x < -self.x_threshold
                or x > self.x_threshold
                or theta < -self.theta_threshold_radians
                or theta > self.theta_threshold_radians
            )
        elif self.task == 'swingup':
            done = bool(
                x < -self.x_threshold
                or x > self.x_threshold
            )
        else:
            raise NotImplementedError
        
        if not done:
            if self.smooth_reward:
                if self.task == 'balance':
                    # theta = np.abs(np.clip(theta, -self.theta_threshold_radians, self.theta_threshold_radians))
                    # x = np.abs(np.clip(x, -self.x_threshold, self.x_threshold))
                    # reward = (2 - theta / self.theta_threshold_radians) * (2 - x / self.x_threshold) - 1
                    reward_theta = np.cos((theta/self.theta_threshold_radians)*(np.pi/2.0))
                    reward_x = np.cos((x/self.x_threshold)*(np.pi/2.0))
                    reward = reward_theta*reward_x
                elif self.task == 'swingup':
                    reward_theta = (np.cos(theta)+1.0)/2.0
                    reward_x = np.cos((x/self.x_threshold)*(np.pi/2.0))
                    reward = reward_theta*reward_x
                self.episode_reward += reward
            else:
                reward = 1.0
        elif self.steps_beyond_done is None:
            # either pole fell or xlimit exceeded
            self.steps_beyond_done = 0
            reward = 0.0
            self.episode_reward += reward
            self.time_step_metric.log(self.step_num)
            self.reward_metric.log(self.episode_reward)
        else:
            self.steps_beyond_done += 1
            reward = 0.0
            self.episode_reward += reward
        
        self.step_num += 1
        if self.step_num >= self.max_ep_len:
            done = True
        else:
            if self.fixed_length_episode:
                done = False
        
        # if visiting terminal states before enough steps, 
        #   then repeat the last states and continue to sample actions
        if self.steps_beyond_done is not None and self.steps_beyond_done > 0:
            self.state = state

        # return np.array(self.state), reward, done, {'terminal': done or (self.steps_beyond_done is not None and self.steps_beyond_done > 10)}
        return np.array(self.state), reward, done, {'terminal': self.steps_beyond_done is not None and self.steps_beyond_done > 1}

    def reset(self):
        if self.task == 'balance':
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        elif self.task == 'swingup':
            self.state = np.random.normal(loc=np.array([0.0, 0.0, np.pi, 0.0]), scale=np.array([0.2, 0.2, 0.2, 0.2]))

        self.steps_beyond_done = None
        self.step_num = 0
        self.episode_reward = 0
            
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



