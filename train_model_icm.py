import logging
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecEnvWrapper
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import torch
import gymnasium as gym
from stable_baselines3 import PPO
import environment_pyboy_neat_pkmn_yellow_icm as emt
from callback import TrainAndLoggingCallback
from stable_baselines3.common.monitor import Monitor
from icm_module import ICM


class ICMWrapper(VecEnvWrapper):
    def __init__(self, venv, icm):
        super(ICMWrapper, self).__init__(venv)
        self.icm = icm
        self.last_obs = None
        self.last_actions = None

    def step_async(self, actions):
        self.last_actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, info = self.venv.step_wait()
        next_obs = obs  # Current obs are next_obs for the next step

        if self.last_obs is not None and self.last_actions is not None:
            obs_tensor = torch.tensor(self.last_obs).float().to(self.icm.device).permute(0, 3, 1, 2)
            next_obs_tensor = torch.tensor(next_obs).float().to(self.icm.device).permute(0, 3, 1, 2)
            actions_tensor = torch.tensor(self.last_actions).to(self.icm.device)

            # print(f"obs_tensor shape: {obs_tensor.shape}")
            # print(f"next_obs_tensor shape: {next_obs_tensor.shape}")
            # print(f"actions_tensor shape: {actions_tensor.shape}")

            intrinsic_rewards = self.icm.compute_intrinsic_reward(obs_tensor, next_obs_tensor, actions_tensor)
            rewards += intrinsic_rewards.detach().cpu().numpy()

        self.last_obs = next_obs  # Update last_obs with the current obs
        return obs, rewards, dones, info

    def reset(self):
        self.last_obs = None
        self.last_actions = None
        return self.venv.reset()


def make_env(env_class):
    def _init():
        env = env_class()
        env = MaxAndSkipEnv(env, 15)
        return env

    return _init


def train_model(env_class, model_path, time_steps):
    n_envs = 8
    checkpoint_dir = './train/'
    log_dir = './logs/'
    env_class = emt.GbaGame
    env = make_vec_env(make_env(env_class), n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine action size
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_size = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        action_size = env.action_space.shape[0]
    else:
        raise NotImplementedError("Unsupported action space type")

    # Initialize ICM
    icm = ICM(obs_shape=env.observation_space.shape, action_size=action_size, device=device).to(device)
    env = ICMWrapper(env, icm)

    total_time_steps = time_steps
    callback = TrainAndLoggingCallback(check_freq=4096, save_path=checkpoint_dir)

    policy_kwargs = dict(
        net_arch=[dict(pi=[512, 256, 128, 64], vf=[512, 256, 128, 64])]
    )

    model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir,
                verbose=1, gamma=0.99, n_steps=2048, n_epochs=2, batch_size=512, ent_coef=0.03,
                learning_rate=0.00003)

    if model_path:
        pass
        # model.load(model_path)

    model.learn(total_timesteps=total_time_steps, callback=callback)
    env.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model_pth = 'train/end_of_training_run8.zip'
    train_model(emt.GbaGame, model_pth, time_steps=5000000)
