# Wraps reinforcement learning tasks provided by OpenAI Gym

import gym
import random

import networks

class Task(object):
    def __init__(self, config):
        self.config = config
        self.name = config['gym_env']
        self.actions = config['actions']
        self.step_count = 0
        self.env = None

    def create_network(self):
        raise NotImplementedError

    def step(self, action):
        self.step_count += 1
        return self.env.step(self.actions[action]) + (False, )

    def render(self):
        return self.env.render()

    def reset(self):
        return self.env.reset()

    def close(self):
        return self.env.close()

class AtariTask(Task):
    def __init__(self, config):
        super(AtariTask, self).__init__(config)
        if config['show_game']:
            self.env = gym.make(self.name, render_mode = "human", frameskip = 1)
        else:
            self.env = gym.make(self.name, frameskip = 1)
        self.env = gym.wrappers.AtariPreprocessing(self.env, scale_obs=True)
        self.env = gym.wrappers.FrameStack(self.env, num_stack=4, lz4_compress=True)

    def create_network(self):
        return networks.AtariNetwork(len(self.actions))

    def render(self):
        return # Atari games in OpenAI gym don't support this method any more

    def reset(self):
        self.current_lives = None
        return self.env.reset()

    def step(self, action):
        self.step_count += 1
        state, reward, done, info = self.env.step(self.actions[action])
        lives = info['lives']
        dead = (self.current_lives is not None and lives != self.current_lives)
        self.current_lives = lives
        return state, reward, done, info, dead


class CartpoleTask(Task):
    def __init__(self, config):
        super(CartpoleTask, self).__init__(config)
        self.env = gym.make(self.name)
        self.state_len = len(self.env.reset())

    def create_network(self):
        return networks.FCNetwork(self.state_len, len(self.actions))


def make_task(config):

    name = config['task_class'].casefold()

    if name == 'cartpole':
        return CartpoleTask(config)

    if name == 'breakout':
        return AtariTask(config)

    if name == 'pong':
        return AtariTask(config)
