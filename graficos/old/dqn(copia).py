import os
import random
import matplotlib.pyplot as plt
from IPython.display import display, clear_output # type: ignore

from collections import namedtuple, deque
import itertools as it
from time import time, sleep

import numpy as np
import skimage
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import vizdoom as vzd

# Q-learning
LEARNING_RATE = 0.01
GAMMA = 0.99 
EPSILON=1
EPSILON_DECAY=0.999
BUFFER_SIZE=10000

# NN learning settins
BATCH_SIZE=40

# Otros parametros
FRAME_REPEAT = 4
RESOLUTION = (60, 45)
EPISODES_TO_TRAIN = 15
EPISODES_TO_PLAY = 100

# Paths Scenarios
scenario = 'defend_the_center'
config_file_path = os.path.join(os.path.dirname(__file__), "..", "ViZDoom", "scenarios", f"{scenario}.cfg")
model_savefile = os.path.join(os.path.dirname(__file__), "..", "models", f"{scenario}.pth")

# Flags
save_model = True
load_model = False
skip_learning = False
window_visible = False

# Uses GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    print("Using GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")

def create_game():
    print("Creating game...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(window_visible)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()
    print("Game created.")

    return game

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

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0], 32, kernel_size=7, stride=1).to(device)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2).to(device)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=1).to(device)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2).to(device)
        fc_input_dims = self.conv_output_dims(state_dim)
        self.fc1 = nn.Linear(fc_input_dims, 800).to(device)
        self.fc2 = nn.Linear(800, action_dim).to(device)

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
        state = torch.zeros(1, *input_dims).to(device)
        dims = self.conv1(state)
        dims = self.pool1(dims)
        dims = self.conv2(dims)
        dims = self.pool2(dims)
        return int(np.prod(dims.size()[1:]))
    
class DQNAgent:
    def __init__(
        self, 
        input_shape, 
        num_actions, 
        lr, 
        gamma, 
        epsilon, 
        epsilon_decay, 
        buffer_size, 
        batch_size
    ):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = ReplayMemory(buffer_size)
        self.model = DQN(input_shape, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size

    def preprocess(self, img):
        img = skimage.transform.resize(img, RESOLUTION)
        img = np.stack((img,)*3, axis=-1)
        img = np.moveaxis(img, 2, 0)
        img = img.astype(np.float32)
        #img = np.expand_dims(img, axis=0)
        return img

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(device)
            state = state.unsqueeze(0)
            with torch.no_grad():
                return self.model(state).max(1)[1].view(1, 1).item()

    def remember(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.tensor(np.array([s for s in batch.next_state if s is not None]), dtype=torch.float32).to(device)
        
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)

        state_action_values = self.model(state_batch).gather(1, action_batch.view(-1, 1))

        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon *= self.epsilon_decay

    def save_model(self):
        torch.save(self.model.state_dict(), model_savefile)
        #print("Model saved to: ", model_savefile)

    def load_model(self):
        self.model.load_state_dict(torch.load(model_savefile))
        self.model.eval()
        print("Model loaded from: ", model_savefile)


def main():
    game = create_game()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    agent = DQNAgent(
        input_shape=(3, RESOLUTION[0], RESOLUTION[1]),
        num_actions=len(actions),
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE
    )

    if load_model:
        agent.load_model()

    # If training, train the model
    if not skip_learning:
        start_time = time()
        episodes = EPISODES_TO_TRAIN
        episode_rewards = []

        fig, ax = plt.subplots()
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        line, = ax.plot([], [])

        print("Training...")
        for episode in range(episodes):
            game.new_episode()
            state = agent.preprocess(game.get_state().screen_buffer)
            total_reward = 0
            for step in it.count():
                action = agent.select_action(state)
                reward = game.make_action(actions[action], FRAME_REPEAT)
                done = game.is_episode_finished()
                total_reward += reward
                next_state = agent.preprocess(game.get_state().screen_buffer) if not done else None
                agent.remember(state, action, next_state, reward)
                state = next_state

                agent.train()

                if done:
                    episode_rewards.append(total_reward)
                    print(f"Episode: {episode}, Reward: {total_reward}")
                    break
            
            if save_model:
                agent.save_model()

            # Actualizar el gráfico
            line.set_xdata(range(len(episode_rewards)))
            line.set_ydata(episode_rewards)
            ax.relim()
            ax.autoscale_view()
            plt.savefig(f"{scenario}.png")

        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    # If not training, play the game
    print("Playing game...")
    for _ in range(EPISODES_TO_PLAY):
        game.new_episode()
        while not game.is_episode_finished():
            state = agent.preprocess(game.get_state().screen_buffer)
            best_action_index = agent.select_action(state)

            game.set_action(actions[best_action_index])
            for _ in range(FRAME_REPEAT):
                game.advance_action()

        # Sleep between episodes
        sleep(0.5)
        score = game.get_total_reward()
        print("Total score: ", score)
    game.close()

if __name__ == "__main__":
    main()