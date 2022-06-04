# Wraps reinforcement learning tasks provided by OpenAI Gym

import gym
import random

import networks

class TaskInterface(object):
    def __init__(self, config):
        return

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class GymTask(TaskInterface):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.name = config['gym_env']
        self.actions = config['actions']
        self.step_count = 0
        self.env = None

    def step(self, action):
        self.step_count += 1
        return self.env.step(self.actions[action]) + (False, )

    def render(self):
        if 'show_game' in self.config and self.config['show_game']:
            return self.env.render()
        else:
            return

    def reset(self):
        return self.env.reset()

    def close(self):
        return self.env.close()

class CartpoleTask(GymTask):
    def __init__(self, config):
        super().__init__(config)
        self.env = gym.make(self.name)
        self.state_len = len(self.env.reset())

class AtariTask(GymTask):
    def __init__(self, config):
        super().__init__(config)
        if 'show_game' in config and config['show_game']:
            self.env = gym.make(self.name, render_mode = "human", frameskip = 1)
        else:
            self.env = gym.make(self.name, frameskip = 1)
        self.env = gym.wrappers.AtariPreprocessing(self.env, scale_obs=True)
        self.env = gym.wrappers.FrameStack(self.env, num_stack=4, lz4_compress=True)

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


