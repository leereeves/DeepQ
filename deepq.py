# Implementation of the Deep Q learning algorithm

# Tips for debugging, AKA mistakes I made:

# Normalize the input

# Notice that 4 game frames are stacked into one bundle,
# one action is chosen per bundle (that is, per 4 frames)
# and one parameter update is done per _action_.
# So there's one parameter update per 16 frames.
# That target network frequency is confusing. It's:
# once per 40k frames = 10k actions = 2.5k updates

# The RMSProp optimizer used in the paper is different
# from the RMSProp in PyTorch, but that's ok, Adam works.

# Setting the Q target to zero when a life is lost seems 
# to help, but I've yet to confirm this in every game.
# OpenAI discourages this, but the original paper did this.

import datetime
import math
import numpy as np
import random
import time
import torch
import torch.multiprocessing as mp

from os.path import exists
from torch.utils.tensorboard import SummaryWriter 

import memory

def train(task_class, network_class, strategy_class, config):
    device = torch.device(config['device'])
    policy_network = network_class(config).to(device)
    target_network = network_class(config).to(device)

    rank = 0
    q = DeepQ(rank, task_class, strategy_class, policy_network, target_network, config)
    q.train()
           

class DeepQ(object):
    def __init__(self, rank, task_class, strategy_class, policy_network, target_network, config):
        self.config = config
        self.rank = rank
        self.task = task_class(config)
        self.strategy = strategy_class(config)
        self.policy_network = policy_network
        self.target_network = target_network

        self.update_freq = config['update_freq']

        self.device = torch.device(config['device'])
        self.memory = memory.PrioritizedReplayMemory(config['memory_size'])
        self.alpha = 1e-5
        self.gamma = config['gamma']

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr = self.config['learning_rate'])
        self.loss = torch.nn.SmoothL1Loss(reduction = 'none', beta = 1.0)

        # These two terms do not match Mnih 2015. Should be action_space_size and action_count
        self.action_count = len(self.task.actions)
        self.step_count = 0
        self.update_count = 0

        # Load old weights if they exist, to continue training
        if self.rank == 0:
            filename = self.get_model_filename()
            if(exists(self.get_model_filename())):
                t = torch.load(filename, map_location='cpu')
                if t:
                    print("Resuming training from existing model")
                    self.policy_network.load_state_dict(t)
            
            self.target_network.load_state_dict(self.policy_network.state_dict())


    def get_model_filename(self):
        return self.config['checkpoint_filename']


    def choose_action(self, state):
        action = self.strategy.action_without_prediction(self.action_count, self.episode, self.step_count)

        if action is None:
            state_tensor = torch.tensor(np.asarray(state, dtype = np.float32)).to(self.device)
            state_tensor = state_tensor.unsqueeze(0) # Add a batch dimension of length 1
            q = self.policy_network.forward(state_tensor)
            q = q.squeeze(0) # remove batch dimension
            action = self.strategy.action_with_prediction(self.action_count, self.episode, self.step_count, q)
            self.qs.append(q[action].item())

        return action

    def minibatch_update(self):
        # "a uniform random policy is run for replay_start_size frames before learning starts"
        # That is replay_start_size / action_repeat actions (steps)
        if self.step_count < (self.config['replay_start_size'] / self.config['action_repeat']):
            return

        # "update_freq actions are selected by the agent between successive SGD updates"
        if self.step_count % self.update_freq != 0:
            return

        # Ensure we have enough memory to sample a full batch
        if len(self.memory) < self.config['batch_size']:
            return

        # All requirements are satisfied, it's time for an update
        self.update_count += 1

        # "target_update_freq is the frequency (measured in number of parameter updates) 
        # with which the target network is updated.
        if self.rank == 0 and (self.update_count % self.config['target_update_freq']) == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        indexes, batch, weights = self.memory.sample(self.config['batch_size'])
        states, actions, new_states, rewards, dones = list(zip(*batch))

        # Follow PyTorch's advice:
        # "UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. 
        # Please consider converting the list to a single numpy.ndarray with numpy.array() 
        # before converting to a tensor."
        states = np.asarray(states)
        actions = np.asarray(actions)
        new_states = np.asarray(new_states)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        weights = np.asarray(weights)

        # Now create tensors
        states_batch = torch.tensor(states, dtype = torch.float32).to(self.device)
        new_states_batch = torch.tensor(new_states,dtype = torch.float32).to(self.device)
        actions_batch = torch.tensor(actions, dtype = torch.long).to(self.device)
        rewards_batch = torch.tensor(rewards, dtype = torch.float32).to(self.device)
        dones_batch = torch.tensor(dones, dtype = torch.float32).to(self.device)
        weights_batch = torch.tensor(weights, dtype = torch.float32).to(self.device)

        # Calculate the Bellman equation, setting the value of Q* to zero in states after the task is done
        with torch.no_grad():
            policy_q = self.policy_network(new_states_batch)
            target_q = self.target_network(new_states_batch)
            next_actions = policy_q.argmax(axis = 1)
            next_q = target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards_batch + torch.mul(self.gamma * next_q, (1 - dones_batch))
            #target = rewards_batch + torch.mul(self.gamma * new_q.max(axis = 1).values, (1 - dones_batch))

        # Calculate the network's current predictions
        prediction = self.policy_network.forward(states_batch).gather(1,actions_batch.unsqueeze(1)).squeeze(1)

        # Calculate annealing factor for importance sampling
        # Grows linearly from 0.4 to 1 during exploration as epsilon decreases
        self.beta = 0.4 + (0.6 * min([self.step_count / self.config['final_exploration_step'], 1]))

        # Normalize weights so they only scale the update downwards
        weights_batch = weights_batch / weights_batch.max()

        # Train the network to predict the results of the Bellman equation        
        self.optimizer.zero_grad()
        loss = self.loss(prediction, target)
        loss = loss * weights_batch.pow(self.beta) # importance sampling
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        # Update weights in prioritized replay memory
        with torch.no_grad():
            deltas = (target-prediction).absolute()
        for i in range(self.config['batch_size']):
            self.memory.update_weight(indexes[i], deltas[i].item() + self.alpha)

        return

    def save_model(self):
        torch.save(self.policy_network.state_dict(), self.get_model_filename())

    def train(self):
        # Open Tensorboard log, only in first process
        if self.rank == 0:
            path = "./tensorboard/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            tb_log = SummaryWriter(path)
            #hp = "config: {}".format(str(self.config))
            #log.add_text("Hyperparameters", hp)
            for key, value in self.config.items():
                tb_log.add_text(key, str(value))

        start = time.time()
        scores = []

        for self.episode in range(self.config['max_episodes']):
            # This is the start of an episode
            state = self.task.reset()
            score = 0
            done = 0
            self.qs = [0]
            while not done:
                # This loops through steps (4 frames for Atari, 1 frame for Cartpole)
                self.task.render()
                action = self.choose_action(state)
                new_state, reward, done, info, dead = self.task.step(action)
                self.step_count += 1
                score += reward
                clipped_reward = np.sign(reward)
                self.memory.store_transition(state, action, new_state, clipped_reward, done or dead)
                self.minibatch_update()
                state = new_state

            scores.append(score)
            moving_average = np.average(scores[-20:])
            elapsed_time = math.ceil(time.time() - start)
            if self.rank == 0: # only the first process will save, print, and log

                tb_log.add_scalars(self.config['name'], {'score': scores[-1]}, self.episode)

                print("Time {}. Episode {}. Action {}. Score {:0.0f}. MAvg={:0.1f}. {}. Avg p={:0.2f}. Avg q={:0.2f}".format(
                    datetime.timedelta(seconds=elapsed_time), 
                    self.episode, 
                    self.step_count, 
                    score, 
                    moving_average, 
                    str(self.strategy), 
                    self.memory.tree.get_average_weight(), 
                    np.average(self.qs)))

                if self.episode > 0 and self.episode % 10 == 0:
                    print("Saving model {}".format(self.get_model_filename()))
                    self.save_model()

        self.task.close()
        tb_log.close()

