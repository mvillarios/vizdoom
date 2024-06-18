import os
import itertools as it
import vizdoom as vzd
from dqn_agent import DQNAgent
from utils import create_game, train_agent, play_game, play_recorded_game

# Q-learning settings
LEARNING_RATE = 0.01
GAMMA = 0.99 
BUFFER_SIZE = 10000
EPSILON = 1.0
EPSILON_DECAY_START = 3e5
EPSILON_DECAY_END = 6e5
EPSILON_MIN = 0.1

# NN learning settings
BATCH_SIZE = 40

# Other parameters
FRAME_REPEAT = 4
RESOLUTION = (60, 45)
EPISODES_TO_TRAIN = 20
STEPS_TO_TRAIN = 12e5
EPISODES_TO_PLAY = 100

# Paths Scenarios
scenario = 'defend_the_center'
config_file_path = os.path.join(vzd.scenarios_path, f"{scenario}.cfg")
model_savefile = os.path.join(os.path.dirname(__file__), "..", "models", f"{scenario}.pth")

# Flags
save_model = False
load_model = False

# config
#config = [True, False, False, False] # Train config
#config = [False, True, False, True] # Play config
config = [False, False, True, True] # Play recorded config

train = config[0]
play = config[1]
play_recorded = config[2]
window_visible = config[3]

def main():
    game = create_game(config_file_path, window_visible)
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    agent = DQNAgent(
        input_shape=(3, *RESOLUTION),
        num_actions=len(actions),
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay_start=EPSILON_DECAY_START,
        epsilon_decay_end=EPSILON_DECAY_END,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        model_savefile=model_savefile
    )

    if load_model:
        agent.load_model()

    if train:
        # Print parameters
        print("Parameters:")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Gamma: {GAMMA}")
        print(f"Buffer size: {BUFFER_SIZE}")
        print(f"Epsilon: {EPSILON}")
        print(f"Epsilon decay start: {EPSILON_DECAY_START}")
        print(f"Epsilon decay end: {EPSILON_DECAY_END}")
        print(f"Epsilon min: {EPSILON_MIN}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Stept to tain: {STEPS_TO_TRAIN}")
        train_agent(game, agent, actions, scenario, save_model, STEPS_TO_TRAIN, FRAME_REPEAT, EPSILON_DECAY_START, EPSILON_DECAY_END)

    if play_recorded:
        directorio = "graficos/12-06/12-06-1"
        files = os.listdir(directorio)
        lmp_files = [file for file in files if file.endswith(".lmp")]
        for lmp_file in lmp_files:
            print(f"Playing {lmp_file}")
            path = os.path.join(directorio, lmp_file)
            play_recorded_game(game, path)

    if play:
        play_game(game, agent, actions, EPISODES_TO_PLAY, FRAME_REPEAT)

    game.close()

if __name__ == "__main__":
    main()

