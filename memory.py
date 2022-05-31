# Two implementations of replay memory
#
# class ReplayMemory is a simple flat array from which
# past transistions are uniformly sampled.
#
# class PrioritizedReplayMemory returns transitions with large
# temporal-difference error (|target-prediction|)more frequently,
# inspired by Schaul (2015) https://arxiv.org/abs/1511.05952v4
# but using periodic systematic resets rather than randomization
# to schedule review of past transitions whose target values
# may have changed.

import random

import sumtree

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = [None] * self.capacity
        self.allocated = 0
        self.index = 0

    def __len__(self):
        return self.allocated

    def store_transition(self, state, action, new_state, reward, done, weight):
        self.buffer[self.index] = (state, action, new_state, reward, done)
        if (self.allocated + 1) < self.capacity:
            self.allocated += 1
            self.index += 1
        else:
            self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        indexes = random.sample(range(self.allocated), k=batch_size)
        samples = [self.buffer[i] for i in indexes]
        return indexes, samples

    def update_weight(self, index, weight):
        return

    def set_all_weights(self, weight):
        return

class PrioritizedReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.tree = sumtree.SumTree(capacity)

    def __len__(self):
        return self.tree.allocated

    def store_transition(self, state, action, new_state, reward, done, weight):
        t = (state, action, new_state, reward, done)
        self.tree.push(t, weight)

    def sample(self, batch_size):
        indexes = []
        samples = []
        while len(indexes) < batch_size:
            r = random.random() * self.tree.get_total_weight()
            i = self.tree.get_index_by_weight(r)
            if i not in indexes:
                t = self.tree.get_data(i)
                indexes.append(i)
                samples.append(t)

        return indexes, samples

    def update_weight(self, index, weight):
        self.tree.set_weight(index, weight)
        return

    def set_all_weights(self, weight):
        for i in range(len(self)):
            self.update_weight(i, weight)