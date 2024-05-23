import environment_pyboy_neat_pkmn_yellow_mlp as emt
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from gymnasium import spaces
import numpy as np
import gymnasium as gym


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


def run_episodes(env_class, model_path, num_episodes=10):
    env = env_class()
    env.max_episodes = 1000000000000
    env = MaxAndSkipEnvDict(env, 15)
    model = emt.PPO.load(model_path, env=env)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action_array, _states = model.predict(obs, deterministic=False)
            action = int(action_array)
            env.render()
            # print(f'Predicted action: {action}')  # Print the predicted action
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if truncated:
                done = True
        print(f'Episode: {episode + 1}, Total Reward: {total_reward}')
    env.close()


if __name__ == "__main__":
    mdl_path = 'train/best_model_274432.zip'
    run_episodes(emt.GbaGame, mdl_path, num_episodes=1)
