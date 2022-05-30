# Implementation of the Deep Q learning algorithm
# without task specific details, which are in tasks.py

import math
import numpy as np
import random
import time
import torch
from datetime import timedelta
from os.path import exists

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = [None] * self.capacity
        self.allocated = 0
        self.index = 0

    def store_transition(self, state, action, new_state, reward, done):
        self.buffer[self.index] = (state, action, new_state, reward, done)
        if (self.allocated + 1) < self.capacity:
            self.allocated += 1
            self.index += 1
        else:
            self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        ii = random.sample(range(self.allocated), k=batch_size)
        return [self.buffer[i] for i in ii]

class DeepQ(object):
    def __init__(self, task, config):
        self.task = task
        self.config = config
        self.device = torch.device(config['device'])
        self.batch_size = config['batch_size']
        self.memory = ReplayMemory(config['memory_size'])
        self.replay_start_size = config['replay_start_size']

        self.policy_network = self.task.create_network(self.device)
        self.target_network = self.task.create_network(self.device)

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

    def should_explore(self):
        # a uniform random policy is run for this number of frames
        if self.task.step_count < self.replay_start_size:
            return True
        elif self.task.should_explore():
            return True
        else:
            return False

    def choose_action(self, state):
        if self.should_explore():
            action = np.random.randint(0, len(self.task.actions))
        else:
            state_tensor = torch.tensor(np.asarray(state, dtype = np.float32)).to(self.device)
            state_tensor = state_tensor.unsqueeze(0) # Add a batch dimension of length 1
            q = self.policy_network.forward(state_tensor)
            action = torch.argmax(q).item()

        return action

    def minibatch_update(self):
        # a uniform random policy is run for this number of frames before learning starts
        if self.memory.allocated < self.replay_start_size or self.memory.allocated < self.batch_size:
            return

        if (self.task.step_count % self.config['target_update']) == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        batch = self.memory.sample(self.batch_size)
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

        # Now create tensors
        states_batch = torch.tensor(states, dtype = torch.float32).to(self.device)
        new_states_batch = torch.tensor(new_states,dtype = torch.float32).to(self.device)
        actions_batch = torch.tensor(actions, dtype = torch.long).to(self.device)
        rewards_batch = torch.tensor(rewards, dtype = torch.float32).to(self.device)
        dones_batch = torch.tensor(dones, dtype = torch.float32).to(self.device)

        # Calculate the Bellman equation, setting the value of Q* to zero in states after the task is done
        with torch.no_grad():
            policy_q = self.policy_network(new_states_batch)
            target_q = self.target_network(new_states_batch)
            next_actions = policy_q.argmax(axis = 1)
            next_q = target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards_batch + torch.mul(self.task.gamma * next_q, (1 - dones_batch))
            #target = rewards_batch + torch.mul(self.task.gamma * new_q.max(axis = 1).values, (1 - dones_batch))

        # Calculate the network's current predictions
        prediction = self.policy_network.forward(states_batch).gather(1,actions_batch.unsqueeze(1)).squeeze(1)

        # Train the network to predict the results of the Bellman equation
        self.policy_network.optimizer.zero_grad()
        loss = self.policy_network.loss(prediction, target)
        loss.backward()
        self.policy_network.optimizer.step()

        return

    def save_model(self):
        torch.save(self.policy_network.state_dict(), self.get_model_filename())

    def train(self):

        start = time.time()
        scores = []
        for self.episode in range(self.task.num_episodes):
            state = self.task.reset()
            score = 0
            done = 0
            while not done:
                self.task.render()
                action = self.choose_action(state)
                new_state, reward, done, info = self.task.step(action)
                score += reward
                self.memory.store_transition(state, action, new_state, reward, done)
                self.minibatch_update()
                state = new_state

            scores.append(score)
            t = math.ceil(time.time() - start)
            print("Time {}. Episode {}. Step {}. Score {}. MAvg={}. Epsilon={}".format(
                timedelta(seconds=t), self.episode, self.task.step_count, score, np.average(scores[-10:]), self.task.epsilon))

            if self.episode > 0 and self.episode % 10 == 0:
                print("Saving model {}".format(self.get_model_filename()))
                self.save_model()


        self.task.close()

