# Implementation of the Deep Q learning algorithm
# without task specific details, which are in tasks.py

# References
#
# Useful empirical tips at:
# https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756

import datetime
import math
import numpy as np
import random
import time
import torch
from os.path import exists
from torch.utils.tensorboard import SummaryWriter 

import memory

class DeepQ(object):
    def __init__(self, task, config):
        self.task = task
        self.config = config
        self.device = torch.device(config['device'])
        self.batch_size = config['batch_size']
        self.memory = memory.PrioritizedReplayMemory(config['memory_size'])
        self.replay_start_size = config['replay_start_size']
        self.memory_alpha = 0.1 # somewhat less than 1 so rewards are prioritized
        self.gamma = config['gamma']

        self.num_episodes = config['max_episodes']
        self.initial_exploration = config['initial_exploration']
        self.final_exploration = config['final_exploration']
        self.final_exploration_step = config['final_exploration_step']
        self.epsilon = self.initial_exploration

        self.lr = config['learning_rate']
        self.policy_network = self.task.create_network().to(self.device)
        self.target_network = self.task.create_network().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr = self.lr)
        self.loss = torch.nn.SmoothL1Loss(reduction = 'none', beta = 1.0)

        # Load old weights if they exist, to continue training
        filename = self.get_model_filename()
        if(exists(self.get_model_filename())):
            t = torch.load(filename, map_location='cpu')
            if t:
                print("Resuming training from existing model")
                self.policy_network.load_state_dict(t)
        
        self.target_network.load_state_dict(self.policy_network.state_dict())


    def get_model_filename(self):
        return self.config['checkpoint_filename']
        #return "".join(c for c in self.task.name if c.isalnum()) + ".pt"


    def compute_epsilon(self):
        # a uniform random policy is run for this number of frames
        if self.task.step_count < self.replay_start_size:
            self.epsilon = 1
        elif self.episode % self.config['eval_episode_freq'] == 0:
            self.epsilon = 0.01
        elif self.task.step_count > self.final_exploration_step:
            self.epsilon = self.final_exploration
        else:
            self.epsilon = self.initial_exploration + \
                (self.task.step_count / self.final_exploration_step) * (self.final_exploration - self.initial_exploration)


    def choose_action(self, state):
        self.compute_epsilon()
        if random.random() < self.epsilon:
            action = np.random.randint(0, len(self.task.actions))
        else:
            state_tensor = torch.tensor(np.asarray(state, dtype = np.float32)).to(self.device)
            state_tensor = state_tensor.unsqueeze(0) # Add a batch dimension of length 1
            q = self.policy_network.forward(state_tensor)
            max, index = torch.max(q, dim=1)
            action = index.item()
            self.qs.append(max.item())

        return action

    def minibatch_update(self):
        # a uniform random policy is run for this number of frames before learning starts
        if len(self.memory) < self.replay_start_size or len(self.memory) < self.batch_size:
            return

        if (self.task.step_count % self.config['target_update']) == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        indexes, batch, weights = self.memory.sample(self.batch_size)
        states, actions, new_states, rewards, dones = list(zip(*batch)) # unzip the tuples

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
        beta = 0.4 + (0.6 * np.min([self.task.step_count / self.final_exploration_step, 1]))

        # Normalize weights so they only scale the update downwards
        weights_batch = weights_batch / weights_batch.max()

        # Train the network to predict the results of the Bellman equation        
        self.optimizer.zero_grad()
        loss = self.loss(prediction, target)
        loss = loss * weights_batch.pow(beta) # importance sampling
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        # Update weights in prioritized replay memory
        with torch.no_grad():
            deltas = (target-prediction).absolute()
        for i in range(self.batch_size):
            self.memory.update_weight(indexes[i], deltas[i].item() + self.memory_alpha)

        return

    def save_model(self):
        torch.save(self.policy_network.state_dict(), self.get_model_filename())

    def train(self):
        # Open Tensorboard log
        path = "./tensorboard/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log = SummaryWriter(path)
        hp = "lr: {} bsize: {} gamma: {}".format(self.lr, self.batch_size, self.gamma)
        log.add_text("Hyperparameters", hp)
        
        start = time.time()
        scores = []

        for self.episode in range(self.num_episodes):
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
                score += reward
                clipped_reward = np.sign(reward)
                self.memory.store_transition(state, action, new_state, clipped_reward, done or dead)
                self.minibatch_update()
                state = new_state

            scores.append(score)
            moving_average = np.average(scores[-100:])
            elapsed_time = math.ceil(time.time() - start)
            print("Time {}. Episode {}. Step {}. Score {:0.0f}. MAvg={:0.1f}. ε={:0.2f}. Avg p={:0.2f}. Avg q={:0.2f}".format(
                datetime.timedelta(seconds=elapsed_time), 
                self.episode, 
                self.task.step_count, 
                score, 
                moving_average, 
                self.epsilon, 
                self.memory.tree.get_average_weight(), 
                np.average(self.qs)))

            if self.episode > 0 and self.episode % 10 == 0:
                print("Saving model {}".format(self.get_model_filename()))
                self.save_model()

            log.add_scalars('episode', {'score': scores[-1], 'score_average': moving_average}, self.episode)

        self.task.close()
        log.close()

