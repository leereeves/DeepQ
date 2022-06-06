import math
import numpy as np
import random
import torch

class StrategyInterface(object):
    def __init__(self, config):
        self.config = config

    def action_without_prediction(self, action_count, episode, step):
        raise NotImplementedError

    def action_with_prediction(self, action_count, episode, step, prediction):
        raise NotImplementedError

class EpsilonGreedyStrategy(StrategyInterface):
    def __init__(self, config):
        super().__init__(config)
        self.epsilon = self.config['initial_exploration']

    def __str__(self):
        return f"ε={self.epsilon:0.2f}"

    def action_without_prediction(self, action_count, episode, step):
        # a uniform random policy is run for this number of frames
        if step < self.config['replay_start_size']:
            self.epsilon = 1
        elif episode % self.config['eval_episode_freq'] == 0:
            self.epsilon = 0.01
        elif step > self.config['final_exploration_step']:
            self.epsilon = self.config['final_exploration']
        else:
            self.epsilon = self.config['initial_exploration'] + \
                (step / self.config['final_exploration_step']) * \
                (self.config['final_exploration'] - self.config['initial_exploration'])

        if random.random() < self.epsilon:
            action = np.random.randint(action_count)
        else:
            action = None # indicates caller must call action_with_prediction

        return action

    def action_with_prediction(self, action_count, episode, step, prediction):
        action = torch.argmax(prediction).item()
        return action

