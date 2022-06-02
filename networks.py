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
        # Output scaling (see comment in forward())
        # self.output_scale = 0.01

        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size = 8, stride = 4, dtype=torch.float32)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2, dtype=torch.float32)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1, dtype=torch.float32)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512, dtype=torch.float32)
        self.fc5 = torch.nn.Linear(512, self.n_actions, dtype=torch.float32)

        self.init_weights()

        self.optimizer = Adam(self.parameters(), lr = learning_rate)
        #self.loss = torch.nn.MSELoss()
        self.loss = torch.nn.SmoothL1Loss()
        self.to(self.device)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        # Without flattening the input tensor the following line gives the error:
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (28672x7 and 3136x512)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc4(x))
        x = self.fc5(x)
        # This network initially generates random values whose variance is close to 1,
        # so the rewards are on the same order of magnitude as the randomness
        # and it is hard to learn from that information. To solve this problem,
        # I scale the output so the variance of the random outputs is much less than 1
        # return x * self.output_scale
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0.0)
