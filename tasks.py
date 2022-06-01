# Wraps reinforcement learning tasks provided by OpenAI Gym

import gym
import random

import networks

class Task(object):
    def __init__(self, config):
        self.config = config
        self.name = config['gym_env']
        self.actions = config['actions']
        self.num_episodes = config['max_episodes']
        self.initial_exploration = config['initial_exploration']
        self.final_exploration = config['final_exploration']
        self.final_exploration_step = config['final_exploration_step']
        self.gamma = config['gamma']
        self.lr = config['learning_rate']

        self.epsilon = self.initial_exploration
        self.step_count = 0
        self.env = None

    def create_network(self, device):
        return None

    def step(self, action):
        self.step_count += 1
        return self.env.step(self.actions[action])

    def render(self):
        return self.env.render()

    def reset(self):
        return self.env.reset()

    def close(self):
        return self.env.close()

    def should_explore(self):
        if self.step_count > self.final_exploration_step:
            self.epsilon = self.final_exploration
        else:
            self.epsilon = self.initial_exploration + \
                (self.step_count / self.final_exploration_step) * (self.final_exploration - self.initial_exploration)

        return random.random() < self.epsilon

class AtariTask(Task):
    def __init__(self, config):
        super(AtariTask, self).__init__(config)
        if config['show_game']:
            self.env = gym.make(self.name, render_mode = "human", frameskip = 1)
        else:
            self.env = gym.make(self.name, frameskip = 1)
        self.env = gym.wrappers.AtariPreprocessing(self.env, scale_obs=True)
        self.env = gym.wrappers.FrameStack(self.env, num_stack=4, lz4_compress=True)

    def create_network(self, device):
        return networks.AtariNetwork(len(self.actions), self.lr, device)

    def render(self):
        return # Atari games in OpenAI gym don't support this method any more


class CartpoleTask(Task):
    def __init__(self, config):
        super(CartpoleTask, self).__init__(config)
        self.env = gym.make(self.name)
        self.state_len = len(self.env.reset())

    def create_network(self, device):
        return networks.FCNetwork(self.state_len, len(self.actions), self.lr, device)


def make_task(config):

    name = config['task_class'].casefold()

    if name == 'cartpole':
        return CartpoleTask(config)

    if name == 'breakout':
        return AtariTask(config)

    if name == 'pong':
        return AtariTask(config)
