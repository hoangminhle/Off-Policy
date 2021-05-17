import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from rl_nexus.utils.metric import Metric

# def one_hot_encode_loc(x):
#     if x == 0: return [1,0,0,0]
#     elif x == 1: return [0,1,0,0]
#     elif x == 2: return [0,0,1,0]
#     elif x == 3: return [0,0,0,1]
#     else: print('Something wrong', x)

def one_hot_encode(x):
    if x == 0: return [1,0,0,0,0]
    elif x == 1: return [0,1,0,0,0]
    elif x == 2: return [0,0,1,0,0]
    elif x == 3: return [0,0,0,1,0]
    elif x == 4: return [0,0,0,0,1] 
def binary_encode(x):
    if x == 0: return [0,0,0,0]
    elif x == 1: return [0,0,0,1]
    elif x == 2: return [0,0,1,0]
    elif x == 3: return [0,0,1,1]
    elif x == 4: return [0,1,0,0]
    elif x == 5: return [0,1,0,1]
    elif x == 6: return [0,1,1,0]
    elif x == 7: return [0,1,1,1]
    elif x == 8: return [1,0,0,0]
    elif x == 9: return [1,0,0,1]
    elif x == 10: return [1,0,1,0]
    elif x == 11: return [1,0,1,1]
    elif x == 12: return [1,1,0,0]
    elif x == 13: return [1,1,0,1]
    elif x == 14: return [1,1,1,0]
    elif x == 15: return [1,1,1,1]
class Taxi_Env(gym.Env):
    def __init__(self, spec_tree):
        self.max_ep_len = spec_tree['max_ep_len']
        self.length = spec_tree['length']
        self.action_space = spaces.Discrete(6)
        self.fixed_length_episode = spec_tree['fixed_length_episode']
        self.discount_factor = 0.99

        #high = np.array([16]*4)
        #low = np.array([-16.]*4)
        self.n_state = (self.length**2)*16*5
        # self.encoder = np.eye(self.n_state)
        # self.status_encoder = np.eye(5)
        # self.passenger_encoder = np.eye(16)
        # high = np.array([1.]*self.n_state)
        # low = np.array([-1.]*self.n_state)
        # self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # self.observation_space = spaces.Discrete(self.n_state)
        # high = np.array([1.]*31)
        # low = np.array([-1.]*31)
        self.zero = np.zeros((self.length, self.length)).astype(np.float32)
        # self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(low = -1.0, high=1.0, shape = (3,5,5), dtype = np.float32)
        self.state = None
        self.seed()
        self.episode_reward = None
        self.reset()
        self.reward_metric = Metric(
            short_name='rews',
            long_name='trajectory reward',
            formatting_string='{:5.1f}',
            higher_is_better=True)
        self.metrics = [self.reward_metric]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.t = 0
        length = self.length
        self.x = np.random.randint(length)
        self.y = np.random.randint(length)
        self.possible_passenger_loc = [(0,0), (0,length-1), (length-1,0), (length-1, length-1)]
        self.passenger_status = np.random.randint(16)
        self.taxi_status = 4

        # self.state = self.start_state.copy()
        # self.state = self.state_encoding()
        self.episode_reward = 0
        return self.observe()

    def state_encoding(self):
        length = self.length
        return self.taxi_status + (self.passenger_status + (self.x * length + self.y) * 16) * 5

    # def state_decoding(self, state):
    #     length = self.length
    #     taxi_status = state % 5
    #     state = state // 5
    #     passenger_status = state % 16
    #     state = state // 16
    #     y = state % length
    #     x = state // length
    #     return x,y,passenger_status,taxi_status

    def is_done(self):
        """Check if we've finished the episode."""
        return True if self.t >= self.max_ep_len else False

    def observe(self):
        loc_channel = np.array(self.zero)
        loc_channel[self.x, self.y] = 1.0
        passenger_channel = np.array(self.zero)
        for i in range(4):
            if self.passenger_status & (1<<i):
                x,y = self.possible_passenger_loc[i]
                passenger_channel[x,y] = 1.0
        status_channel = np.array(self.zero)
        if self.taxi_status != 4:
            x,y = self.possible_passenger_loc[self.taxi_status]
            status_channel[x,y] = 1.0
        observation = np.stack((loc_channel, passenger_channel, status_channel))
        return observation

        # """Return current state."""
        # # return self.state_decoding(self.state)
        # # return np.array([self.x/5, self.y/5, self.passenger_status/16, self.taxi_status/5])
        # # return np.array([self.x, self.y, self.passenger_status, self.taxi_status]) / 15
        # #x = one_hot_encode(self.x)
        # #y = one_hot_encode(self.y)
        # #passenger_status = binary_encode(self.passenger_status)
        # #taxi_status = one_hot_encode(self.taxi_status)
        # # print(x,y, passenger_status, taxi_status)
        # # return np.array(x+y+passenger_status+taxi_status)
        # #return np.array(x+y+taxi_status+passenger_status)
        # # return np.concatenate((self.status_encoder[self.x,:], self.status_encoder[self.y,:], self.status_encoder[self.taxi_status,:], self.passenger_encoder[self.passenger_status,:]))
        # state = self.state_encoding()
        # return self.encoder[state,:]
        # # return np.array([state])
    def render(self):
        MAP = []
        length = self.length
        for i in range(length):
            if i == 0:
                MAP.append('-'*(3*length+1))
            MAP.append('|' + '  |' * length)
            MAP.append('-'*(3*length+1))
        MAP = np.asarray(MAP, dtype = 'c')
        if self.taxi_status == 4:
            MAP[2*self.x+1, 3*self.y+2] = 'O'
        else:
            MAP[2*self.x+1, 3*self.y+2] = '@'
        for i in range(4):
            if self.passenger_status & (1<<i):
                x,y = self.possible_passenger_loc[i]
                MAP[2*x+1, 3*y+1] = 'a'
        for line in MAP:
            print(b''.join(line))
        if self.taxi_status == 4:
            print('Empty Taxi')
        else:
            x,y = self.possible_passenger_loc[self.taxi_status]
            print('Taxi destination:({},{})'.format(x,y))

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.t += 1
        reward = -0.05
        length = self.length
        if action == 0:
            if self.x < self.length - 1:
                self.x += 1
        elif action == 1:
            if self.y < self.length - 1:
                self.y += 1
        elif action == 2:
            if self.x > 0:
                self.x -= 1
        elif action == 3:
            if self.y > 0:
                self.y -= 1
        elif action == 4:	# Try to pick up
            for i in range(4):
                x,y = self.possible_passenger_loc[i]
                if x == self.x and y == self.y and(self.passenger_status & (1<<i)):
                    # successfully pick up
                    self.passenger_status -= 1<<i
                    self.taxi_status = np.random.randint(4)
                    while self.taxi_status == i:
                        self.taxi_status = np.random.randint(4)
        elif action == 5:
            if self.taxi_status < 4:
                x,y = self.possible_passenger_loc[self.taxi_status]
                if self.x == x and self.y == y:
                    reward = 1.0
                    #print('success')
                self.taxi_status = 4
        self.change_passenger_status()
        # return self.state_encoding(), reward, self.is_done(), {}
        self.episode_reward += reward*self.discount_factor**self.t
        if self.is_done():
            self.reward_metric.log(self.episode_reward)
        return self.observe(), reward, self.is_done(), {}

    def change_passenger_status(self):
        p_generate = [0.3, 0.05, 0.1, 0.2]
        p_disappear = [0.05, 0.1, 0.1, 0.05]
        for i in range(4):
            if self.passenger_status & (1<<i):
                if np.random.rand() < p_disappear[i]:
                    self.passenger_status -= 1<<i
            else:
                if np.random.rand() < p_generate[i]:
                    self.passenger_status += 1<<i
    def debug(self):
        self.reset()
        while True:
            self.render()
            action = input('Action:')
            if action > 5 or action < 0:
                break
            else:
                _, reward = self.step(action)
                print(reward)
