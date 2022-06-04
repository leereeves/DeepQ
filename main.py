import json
import sys
import torch

import networks
import tasks

import deepq


def get_builtin(name, config):

    name = name.casefold()

    if name == 'cartpole':
        task_class = tasks.CartpoleTask
        nn_class   = networks.FCNetwork
        config['state_len'] = 4

    elif name == 'breakout' or name == 'pong':
        task_class = tasks.AtariTask
        nn_class   = networks.AtariNetwork

    return task_class, nn_class, config

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

    task_class, nn_class, config = get_builtin(task_name, config)
    deepq.run_server(task_class, nn_class, config)
