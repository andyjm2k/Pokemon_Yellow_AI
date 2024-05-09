import pickle
import neat
import os
import numpy as np
import environment_pyboy_neat as emt  # Assuming this is your environment

# Load the configuration.
config_path = os.path.join('config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# Load the saved winner.
with open('winner_100.pkl', 'rb') as input_file:
    winner = pickle.load(input_file)

# Recreate the network.
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)


def preprocess_observation(obs):
    # Assuming obs is a NumPy array of shape (height, width, num_frames)
    # Flatten the observation to a 1D array
    flattened_obs = obs.flatten()

    # Optionally, normalize the pixel values from 0-255 to 0-1
    normalized_obs = flattened_obs / 255.0

    return normalized_obs


# Define a function to run simulations or tests with the loaded network.
def run_with_winner(net, env, steps=100000, render=True):
    for episode in range(1):  # Run for 20 episodes for example
        obs, _ = env.reset()  # Extract observation from tuple
        total_reward = 0
        for step in range(steps):
            # Preprocess the observation
            processed_obs = preprocess_observation(obs)

            # Activate the network with the processed observation
            outputs = net.activate(processed_obs)

            # Decide on an action based on the network output
            action = np.argmax(outputs)
            obs, reward, done, trunc, _ = env.step(action)  # Extract new observation from tuple
            total_reward += reward
            if render:
                env.render()
            if done:
                break
        print(f"Episode: {episode}, Total reward: {total_reward}")



# Assuming you have an instance of your environment.
env = emt.GbaGame()
run_with_winner(winner_net, env)
