import json
import sys
import torch

from tasks import make_task
from deepq import DeepQ

def main():

    if len(sys.argv) < 2:
        print("Usage: main <config file>")
        exit()

    with open(sys.argv[1]) as fp:
        config = json.load(fp)

    if 'device' in config:
        print("Training on " + config['device'])
    elif torch.cuda.is_available():
        print("Training on GPU.")
        config['device'] = 'cuda:0'
    else:
        print("No CUDA device found, or CUDA is not installed. Training on CPU.")
        config['device'] = 'cpu'

    task = make_task(config)

    q = DeepQ(task, config)
    q.train()

    task.close()


if __name__=="__main__":
    main()