import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from dqn_model import DQN
from collections import namedtuple, deque
import skimage

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    
    def push(self, *args):
        self.memory.append(self.Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(
        self, 
        input_shape, 
        num_actions, 
        lr, 
        gamma, 
        epsilon, 
        epsilon_decay, # Decaimiento del epsilon
        epsilon_min, # Epsilon mínimo
        epsilon_decay_start, # Pasos para que empiece a decaer el epsilon
        epsilon_decay_end, # Pasos para que termine de decaer el epsilon
        buffer_size, 
        batch_size,
        model_savefile,
    ):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_decay_start = epsilon_decay_start
        self.epsilon_decay_end = epsilon_decay_end
        self.memory = ReplayMemory(buffer_size)
        self.model = DQN(input_shape, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.model_savefile = model_savefile
        self.steps = 0

    def preprocess(self, img):
        img = skimage.transform.resize(img, (self.input_shape[1], self.input_shape[2]))
        img = skimage.color.gray2rgb(img)
        img = np.moveaxis(img, -1, 0)
        img = img.astype(np.float32)
        return img

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.model.device)
            state = torch.unsqueeze(state, 0)
            with torch.no_grad():
                return self.model(state).max(1)[1].view(1, 1).item()

    def remember(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.model.device, dtype=torch.bool)
        non_final_next_states = torch.tensor(np.array([s for s in batch.next_state if s is not None]), dtype=torch.float32).to(self.model.device)
        
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.model.device)
        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=self.model.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.model.device)

        state_action_values = self.model(state_batch).gather(1, action_batch.view(-1, 1))

        next_state_values = torch.zeros(self.batch_size, device=self.model.device)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        if self.steps >= self.epsilon_decay_start:
            decay_steps = self.steps - self.epsilon_decay_start
            total_decay_steps = self.epsilon_decay_end - self.epsilon_decay_start
            self.epsilon = max(self.epsilon_min, self.epsilon - decay_steps * ((self.epsilon - self.epsilon_min) / total_decay_steps))

        return self.epsilon

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_savefile)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_savefile))
        self.model.eval()
