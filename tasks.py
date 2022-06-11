# Wraps reinforcement learning tasks provided by OpenAI Gym

import gym
import numpy as np
import random

import networks

class TaskInterface(object):
    def __init__(self, config):
        return

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class GymTask(TaskInterface):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.name = config['gym_env']
        self.actions = config['actions']
        self.env = None

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            a = action
        else:
            a = np.array([self.actions[action]])
        return self.env.step(a) + (False, )

    def render(self):
        if 'show_game' in self.config and self.config['show_game']:
            return self.env.render()
        else:
            return

    def close(self):
        return self.env.close()

class BasicGymTask(GymTask):
    def __init__(self, config):
        super().__init__(config)
        self.env = gym.make(self.name)

class AtariTask(GymTask):
    def __init__(self, config):
        super().__init__(config)
        if 'show_game' in config and config['show_game']:
            self.env = gym.make(self.name, render_mode = "human", frameskip = 1)
        else:
            self.env = gym.make(self.name, frameskip = 1)
        self.env = gym.wrappers.AtariPreprocessing(self.env, scale_obs=True)
        self.env = gym.wrappers.FrameStack(self.env, num_stack=config['action_repeat'], lz4_compress=True)

    def reset(self):
        self.current_lives = None
        return self.env.reset()

    def render(self):
        return # Atari games in OpenAI gym don't support this method any more

    def step(self, action):
        state, reward, done, info = self.env.step(self.actions[action])
        lives = info['lives']
        dead = (self.current_lives is not None and lives != self.current_lives)
        self.current_lives = lives
        return state, reward, done, info, dead


