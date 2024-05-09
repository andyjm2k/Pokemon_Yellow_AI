from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

import environment_pyboy_neat_pkmn_yellow as emt
import callback as cb
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN


def make_env(env_class):
    def _init():
        env = env_class()
        env = MaxAndSkipEnv(env, 30)
        return env
    return _init


def train_model(env_class, model_path, time_steps):
    n_envs = 1
    checkpoint_dir = './train/'
    log_dir = './logs/'
    env_class = emt.GbaGame  # Replace with your actual environment class
    # Create the vectorized environment
    # env = VecMonitor(SubprocVecEnv([make_env(env_class, i) for i in range(n_envs)]), "logs/TestMonitor")
    env = make_vec_env(make_env(env_class), n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv)
    # model_path = 'path_to_model'  # Replace with your actual model path
    total_time_steps = time_steps
    callback = cb.TrainAndLoggingCallback(check_freq=10250, save_path=checkpoint_dir)

    model = DQN('CnnPolicy', env, tensorboard_log=log_dir,
                learning_rate=0.00003, verbose=1,
                gamma=0.99, learning_starts=1024, train_freq=102400,
                target_update_interval=40960,
                exploration_fraction=0.5)
    if model_path:
        pass
        # model.load(model_path)

    model.learn(total_timesteps=total_time_steps, callback=callback)
    env.close()


if __name__ == '__main__':
    model_pth = 'train/continue_training_model.zip'
    train_model(emt.GbaGame, model_pth, time_steps=5000000)
