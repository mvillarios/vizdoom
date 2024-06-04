import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Elegir device
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(state_dim[0], 32, kernel_size=7, stride=1).to(self.device)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2).to(self.device)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=1).to(self.device)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2).to(self.device)
        fc_input_dims = self.conv_output_dims(state_dim)
        self.fc1 = nn.Linear(fc_input_dims, 800).to(self.device)
        self.fc2 = nn.Linear(800, action_dim).to(self.device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.fc2(x)
        return x
    
    def conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims).to(self.device)
        dims = self.conv1(state)
        dims = self.pool1(dims)
        dims = self.conv2(dims)
        dims = self.pool2(dims)
        return int(np.prod(dims.size()[1:]))
