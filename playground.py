import networks
import deepq
import tasks

class RLPlayground(object):
    def __init__(self):
        super().__init__()
        self.tasks = {}
        self.agents = {}
        self.networks = {}
        self.add_builtins()

    def add_task(self, name, type):
        self.tasks[name] = type

    def get_task(self, name):
        return self.tasks[name]

    def add_agent(self, name, type):
        self.agents[name] = type

    def get_agent(self, name):
        return self.agents[name]

    def add_network(self, name, type):
        self.networks[name] = type

    def get_network(self, name):
        return self.networks[name]

    def add_builtins(self):
        self.add_agent('dqn', deepq.DeepQ)

        self.add_network('fc', networks.FCNetwork)
        self.add_network('mnih2015', networks.AtariNetwork)

        self.add_task('atari', tasks.AtariTask)
        self.add_task('gym', tasks.BasicGymTask)
