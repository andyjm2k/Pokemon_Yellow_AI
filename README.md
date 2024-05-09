# Pokemon Yellow AI Agent

This repository contains a reinforcement learning environment and agent for playing Pokémon Yellow on the Game Boy Advance using the [PyBoy](https://github.com/Baekalfen/PyBoy) emulator. The environment is built using OpenAI's Gymnasium framework and the agent uses [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) for training.

## Features

- Custom Gymnasium environment (`GbaGame`) for playing Pokémon Yellow
- Predefined actions mapped to the game controls
- Frame skipping and frame stacking for optimized training
- Reward functions for game progress tracking
- Agent training using the Proximal Policy Optimization (PPO) algorithm from Stable Baselines 3

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/andyjm2k/Pokemon_Yellow_AI.git
   cd Pokemon_Yellow_AI
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have the required libraries:

   - `mss`
   - `opencv-python`
   - `numpy`
   - `gymnasium`
   - `stable-baselines3`
   - `pyboy`

   You can also install them directly:

   ```bash
   pip install mss opencv-python numpy gymnasium stable-baselines3 pyboy
   ```

3. Download the Pokémon Yellow ROM and place it in the `ROMs` folder.

## Usage

### Training the Agent

You can train the agent using the provided PPO model. Ensure that the ROM is correctly placed and then run:

```bash
   python ./train_model.py
   ```

### Using a Pre-Trained Agent

You can load a pre-trained agent using:

```python
# Load the model
model = PPO.load('path/to/your/model.zip', env=env)

# Test the model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render()
```

### Custom Environment Details

The custom Gymnasium environment `GbaGame` provides the following key features:

- **Action Space:** `Discrete(4)` representing the actions: `['a', 'right', 'left', 'up', 'down']`
- **Observation Space:** `Box(low=0, high=255, shape=(120, 120, 1), dtype=np.uint8)`
- **Reward Function:** Rewards are calculated based on progress in the game, with penalties for inactivity, health loss, and failure to progress.

### Example Code

Here's a snippet showing how to use the environment:

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Create and wrap the environment
env = GbaGame()
env = Monitor(env)

# Initialize the PPO model
model = PPO('CnnPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save('ppo_gba_game')
```

### Testing

To test your agent, you can run:

```bash
python test_agent.py
```

where `test_agent.py` contains:

```python
from stable_baselines3 import PPO

# Load the environment
env = GbaGame()

# Load the model
model = PPO.load('path/to/your/model.zip', env=env)

# Test the model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render()
```

### Acknowledgements

- [PyBoy Emulator](https://github.com/Baekalfen/PyBoy)
- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
- [MSS](https://github.com/BoboTiG/python-mss)

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

Make sure to replace `your-username` with your GitHub username or organization name, and adjust the file paths where necessary.

Let me know if you need any more details added!# Pokemon Yellow AI
This project has a custom gymnasium wrapper for a pyboy emulator environment and uses RL through Stable Baselines 3 to train various models to play Pokemon Yellow.
