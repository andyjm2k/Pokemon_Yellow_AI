from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch.nn as nn
import torch
import numpy as np
import gymnasium as gym

import environment_pyboy_neat_pkmn_yellow_mlp as emt
import callback as cb
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO


class MaxAndSkipEnvDict(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnvDict, self).__init__(env)
        self._skip = skip
        self.env = env
        self.observation_space = env.observation_space

        if isinstance(self.observation_space, spaces.Dict):
            self._obs_buffer = {key: np.zeros((2, *space.shape), dtype=space.dtype)
                                for key, space in self.observation_space.spaces.items()
                                if isinstance(space, spaces.Box)}
        else:
            self._obs_buffer = np.zeros((2, *self.observation_space.shape), dtype=self.observation_space.dtype)

        # print(f"Initialized MaxAndSkipEnvDict with observation space: {self.observation_space}")

    def step(self, action):
        """Repeat action, sum reward, and take max over last observations."""
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            if isinstance(obs, dict):
                for key, value in obs.items():
                    if key in self._obs_buffer:
                        self._obs_buffer[key][1] = value
                        self._obs_buffer[key][0] = np.maximum(self._obs_buffer[key][0], value)
            else:
                self._obs_buffer[1] = obs
                self._obs_buffer[0] = np.maximum(self._obs_buffer[0], obs)
            total_reward += reward
            if done:
                break
        if isinstance(obs, dict):
            max_frame_obs = {key: self._obs_buffer[key][0] if key in self._obs_buffer else value
                             for key, value in obs.items()}
        else:
            max_frame_obs = self._obs_buffer[0]
        return max_frame_obs, total_reward, done, truncated, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, dict):
            for key, value in obs.items():
                if key in self._obs_buffer:
                    self._obs_buffer[key][0] = value
        else:
            self._obs_buffer[0] = obs
        # print(f"Reset MaxAndSkipEnvDict with observation: {obs}")
        return obs


def make_env(env_class, skip=4, **env_kwargs):
    def _init():
        env = env_class(**env_kwargs)
        # print(f"Creating environment with class {env_class} and kwargs {env_kwargs}")
        # print(f"Created MaxAndSkipEnvDict with observation space: {env.observation_space}")
        env = MaxAndSkipEnvDict(env, skip=skip)
        # print(f"Created MaxAndSkipEnvDict with observation space: {env.observation_space}")
        return env
    return _init


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)

        image_space = observation_space['image']
        assert isinstance(image_space, spaces.Box), "Observation space for 'image' must be of type spaces.Box"

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            sample_image = torch.as_tensor(image_space.sample()[None]).float()
            n_flatten = self.cnn(sample_image).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten + 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Convert image observations to float and normalize
        image_tensor = observations['image'].float() / 255.0
        cnn_out = self.cnn(image_tensor)
        combined_input = torch.cat((cnn_out, observations['additional_features']), dim=1)
        return self.fc(combined_input)


def train_model(env_class, model_path, time_steps):
    n_envs = 8
    checkpoint_dir = './train/'
    log_dir = './logs/'
    env_class = emt.GbaGame  # Replace with your actual environment class
    # Create the vectorized environment
    # env = VecMonitor(SubprocVecEnv([make_env(env_class, i) for i in range(n_envs)]), "logs/TestMonitor")
    env = make_vec_env(make_env(env_class, skip=15), n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv)
    # model_path = 'path_to_model'  # Replace with your actual model path
    total_time_steps = time_steps
    callback = cb.TrainAndLoggingCallback(check_freq=4096, save_path=checkpoint_dir)
    # Example architecture: separate networks for policy and value
    policy_kwargs = dict(
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])]
    )

    model = PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir,
                verbose=1, gamma=0.999, n_steps=2048,
                n_epochs=2, batch_size=512, ent_coef=0.03, learning_rate=0.00003)
    if model_path:
        pass
        # model.load(model_path)

    model.learn(total_timesteps=total_time_steps, callback=callback)
    env.close()


if __name__ == '__main__':
    model_pth = 'train/end_of_training_run7.zip'
    train_model(emt.GbaGame, model_pth, time_steps=10000000)

