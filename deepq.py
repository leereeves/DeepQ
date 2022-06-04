# Implementation of the Deep Q learning algorithm
# without task specific details, which are in tasks.py

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

def run_server(task_class, network_class, config):
    # The server creates the shared networks and spawns the scouts
    device = torch.device(config['device'])
    policy_network = network_class(config).to(device)
    target_network = network_class(config).to(device)
    policy_network.share_memory()
    target_network.share_memory()

    network_lock = mp.Lock()
    #q = DeepQ(task_class, policy_network, target_network, config)
    #q.train()
    process_count = 1
    if process_count <= 1:
        rank = 0
        spawn_scout(rank, task_class, policy_network, target_network, config, network_lock)
    else:
        raise NotImplementedError # not working yet
        mp.set_start_method('spawn', force=True)
        p = [None] * process_count
        for rank in range(process_count):
            print("Spawning process {} to train Deep Q network".format(rank))
            p[rank] = mp.Process(target=spawn_scout, args=(rank, task_class, policy_network, target_network, config, network_lock))
            p[rank].start()
        for rank in range(process_count):
            p[rank].join() # wait for process to end
           

def spawn_scout(rank, task_class, policy_network, target_network, config, network_lock):
    q = DeepQ(rank, task_class, policy_network, target_network, network_lock, config)
    q.train()

class DeepQ(object):
    def __init__(self, rank, task_class, policy_network, target_network, network_lock, config):
        self.config = config
        self.rank = rank
        self.network_lock = network_lock
        self.task = task_class(config)
        self.policy_network = policy_network
        self.target_network = target_network
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
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr = self.lr)
        self.loss = torch.nn.SmoothL1Loss(reduction = 'none', beta = 1.0)

        # Load old weights if they exist, to continue training
        if self.rank == 0:
            self.network_lock.acquire()
            filename = self.get_model_filename()
            if(exists(self.get_model_filename())):
                t = torch.load(filename, map_location='cpu')
                if t:
                    print("Resuming training from existing model")
                    self.policy_network.load_state_dict(t)
            
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.network_lock.release()


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
            self.network_lock.acquire()
            q = self.policy_network.forward(state_tensor)
            self.network_lock.release()
            max, index = torch.max(q, dim=1)
            action = index.item()
            self.qs.append(max.item())

        return action

    def minibatch_update(self):
        # a uniform random policy is run for this number of frames before learning starts
        if len(self.memory) < self.replay_start_size or len(self.memory) < self.batch_size:
            return

        if self.rank == 0 and (self.task.step_count % self.config['target_update']) == 0:
            self.network_lock.acquire()
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.network_lock.release()

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

        self.network_lock.acquire()

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

        self.network_lock.release()

        # Update weights in prioritized replay memory
        with torch.no_grad():
            deltas = (target-prediction).absolute()
        for i in range(self.batch_size):
            self.memory.update_weight(indexes[i], deltas[i].item() + self.memory_alpha)

        return

    def save_model(self):
        self.network_lock.acquire()
        torch.save(self.policy_network.state_dict(), self.get_model_filename())
        self.network_lock.release()

    def train(self):
        # Open Tensorboard log, only in first process
        if self.rank == 0:
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
            if self.rank == 0: # only the first process will save, print, and log
                if self.episode > 0 and self.episode % 10 == 0:
                    print("Saving model {}".format(self.get_model_filename()))
                    self.save_model()

                print("Time {}. Episode {}. Step {}. Score {:0.0f}. MAvg={:0.1f}. ε={:0.2f}. Avg p={:0.2f}. Avg q={:0.2f}".format(
                    datetime.timedelta(seconds=elapsed_time), 
                    self.episode, 
                    self.task.step_count, 
                    score, 
                    moving_average, 
                    self.epsilon, 
                    self.memory.tree.get_average_weight(), 
                    np.average(self.qs)))

                log.add_scalars(self.config['name'], {'score': scores[-1], 'score_average': moving_average}, self.episode)

        self.task.close()
        log.close()

