from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

import environment_pyboy as emt
import callback as cb
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C


def make_env(env_class):
    def _init():
        env = env_class()
        env = MaxAndSkipEnv(env, 8)
        return env
    return _init


def train_model(env_class, model_path, time_steps):
    n_envs = 4
    checkpoint_dir = './train/'
    log_dir = './logs/'
    env_class = emt.GbaGame  # Replace with your actual environment class
    # Create the vectorized environment
    # env = VecMonitor(SubprocVecEnv([make_env(env_class, i) for i in range(n_envs)]), "logs/TestMonitor")
    env = make_vec_env(make_env(env_class), n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv)
    # model_path = 'path_to_model'  # Replace with your actual model path
    total_time_steps = time_steps
    callback = cb.TrainAndLoggingCallback(check_freq=4096, save_path=checkpoint_dir)

    # policy_kwargs = dict(
        # net_arch=[dict(pi=[64, 64, 64], vf=[64, 64, 64])]  # Example architecture: separate networks for policy and value
    # )

    model = A2C('CnnPolicy', env, tensorboard_log=log_dir,
                learning_rate=0.00003, verbose=1,
                gamma=0.99, ent_coef=0.01, n_steps=5,
                use_rms_prop=True)
    if model_path:
        pass
        # model.load(model_path)

    model.learn(total_timesteps=total_time_steps, callback=callback)
    env.close()


if __name__ == '__main__':
    model_pth = 'train/continue_training_model.zip'
    train_model(emt.GbaGame, model_pth, time_steps=9000000)
