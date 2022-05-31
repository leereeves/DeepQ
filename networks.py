import torch
from torch.optim import Adam
from torch.nn import Linear, ReLU

# Fully connected network for classic control tasks
class FCNetwork(torch.nn.Module):
    def __init__(self, state_len, n_actions, learning_rate, device):
        super().__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.n_actions = n_actions

        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_len, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions)
        )
        self.optimizer = Adam(self.parameters(), lr = learning_rate)
        self.loss = torch.nn.MSELoss()
        
        self.to(self.device)

    def forward(self, state):
        return self.network(state)

# Convolutional network for Atari games, as described in Mnih 2015
class AtariNetwork(torch.nn.Module):
    def __init__(self, n_actions, learning_rate, device):
        super().__init__()
        self.device = device
        self.n_actions = n_actions
        self.learning_rate = learning_rate

        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size = 8, stride = 4, dtype=torch.float32)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2, dtype=torch.float32)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1, dtype=torch.float32)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512, dtype=torch.float32)
        self.fc5 = torch.nn.Linear(512, self.n_actions, dtype=torch.float32)

        self.init_weights()
        # The initial weights assigned by init_weights tend to result in 
        # Q values that are much too high (Q values should be less than one 
        # until the network learns to play well enough to score points).
        # I adjust for that here by simply reducing the initial weights
        # in the final layer.
        self.fc5.weight = torch.nn.parameter.Parameter(self.fc5.weight / 100)

        self.optimizer = Adam(self.parameters(), lr = learning_rate)
        #self.loss = torch.nn.MSELoss()
        self.loss = torch.nn.SmoothL1Loss()
        self.to(self.device)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        # Without flattening the input tensor the next line gives the error:
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (28672x7 and 3136x512)
        x = torch.nn.functional.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0.0)
