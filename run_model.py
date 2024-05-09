import environment_pyboy_neat_pkmn_yellow as emt


def run_episodes(env_class, model_path, num_episodes=10):
    env = env_class()
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
    mdl_path = 'train/best_model_376832.zip'
    run_episodes(emt.GbaGame, mdl_path, num_episodes=1)
