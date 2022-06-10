import json
import sys
import torch

import deepq
import networks
import strategy
import tasks

from playground import RLPlayground


def get_builtin(name, config):

    name = name.casefold()

    pg = RLPlayground()
    task_class = pg.get_task(config['task_type'])
    nn_class   = pg.get_network(config['network_type'])

    agent_class = deepq.DeepQ

    strategy_name = config['exploration_strategy'].casefold()

    if strategy_name == 'annealing':
        strategy_class = strategy.AnnealingStrategy
    else:
        strategy_class = strategy.EpsilonGreedyStrategy

    return task_class, nn_class, strategy_class, agent_class, config

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: main <task>")
        exit()

    task_name = sys.argv[1]    

    with open(task_name + ".json") as fp:
        config = json.load(fp)

    config['name'] = task_name

    if 'device' in config:
        print("Training on " + config['device'])
    elif torch.cuda.is_available():
        print("Training on GPU.")
        config['device'] = 'cuda:0'
    else:
        print("No CUDA device found, or CUDA is not installed. Training on CPU.")
        config['device'] = 'cpu'

    task_class, nn_class, strategy_class, agent_class, config = get_builtin(task_name, config)
    deepq.train(task_class, nn_class, strategy_class, config)
