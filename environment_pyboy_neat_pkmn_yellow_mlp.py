from mss import mss
import cv2
import numpy as np
from gymnasium import Env, spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
# Import the sb3 monitor for logging
from stable_baselines3.common.monitor import Monitor
import os
from collections import deque
from pyboy import PyBoy, WindowEvent
import random


class GbaGame(Env):
    def __init__(self, max_episodes=400000):
        super().__init__()
        self.frame_stack = deque(maxlen=1)
        self.frame_skip = 1  # Number of frames to skip

        self.image_shape = (120, 120, 1)
        self.num_additional_features = 3  # x, y, and map ID

        # Flattened image size plus additional features
        flattened_image_size = np.prod(self.image_shape)
        observation_size = flattened_image_size + self.num_additional_features

        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8),
            'additional_features': spaces.Box(low=0, high=255, shape=(self.num_additional_features,), dtype=np.float32)
        })
        # Adjust observation space to 3D for CNN compatibility
        # self.observation_space = Box(low=0, high=255, shape=(120, 120, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(6)
        self.cap = mss()
        self.pyboy = PyBoy('ROMs/Pokemon_Yellow.gbc', window_type="headless",
                           window_scale=3, game_wrapper=False)
        self.game_location = {'top': 53, 'left': 0, 'width': 318, 'height': 339}
        self.score_location = {'top': 53, 'left': 65, 'width': 70, 'height': 25}
        self.done_location = {'top': 28, 'left': 21, 'width': 100, 'height': 79}
        self.score_cap = False
        self.penalty_cap = False
        self.reset_game_state()
        self.total_reward = 0
        self.current_step = 0
        self.truncated = False
        self.episode_length = 0
        self.current_score = 0
        self.max_episodes = max_episodes
        self.wait_frames = 1
        self.initial_observation = True
        self.pyboy_counter = 0
        self.level_progress = [-1]
        self.ash_loc_dict = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        self.level_progress_pct = 0.0
        self.ash_is_moving = 0
        self.ash_stuck_counter = 0
        self.battle_stuck_counter = 0
        self.is_battling_fl = False
        self.best_progress = 0.0
        self.best_progress_counter = 0
        self.best_reward = 0.0
        self.flag_reached = False
        self.new_total_hp = 0
        self.first_loc_check = 0
        self.new_enemy_hp = 0
        self.agent_id = random.randint(1, 1000000)
        print('STARTED AGENT: ', self.agent_id)

    def reset_game_state(self):
        self.total_reward = 0
        self.episode_length = 0
        self.current_score = 0
        self.level_progress = [-1]
        self.ash_loc_dict = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        self.level_progress_pct = 0.0
        self.ash_is_moving = 0
        self.ash_stuck_counter = 0
        self.battle_stuck_counter = 0
        self.is_battling_fl = False
        self.penalty_cap = False
        self.score_cap = False
        self.truncated = False
        self.flag_reached = False
        self.new_total_hp = 0
        self.first_loc_check = 0
        self.new_enemy_hp = 0

    def step(self, action):
        total_reward = 0
        done = False
        info = {}
        for _ in range(self.frame_skip):
            if not done:
                self.execute_action(action)
                self.update_frame_stack()
                observation = self.get_observation()
                # observation = self.get_stacked_observation()
                # self.render()
                self.current_step += 1
                self.episode_length += 1
                # print("self.new_enemy_hp = ", self.pyboy.get_memory_value(0xcfe6))
                reward, done = self.calculate_reward_and_done(action)
                truncated = self.truncated
                if truncated:
                    if self.total_reward > self.best_reward:
                        self.best_reward = self.total_reward
                        print("Agent Number: ", self.agent_id)
                        print("Best reward is now: ", self.best_reward)
                    if self.flag_reached:
                        done = True
                # You can aggregate or choose the last 'info' as needed
                info = {}
        # Line below for training
        return observation, reward, done, truncated, info
        # Line below for running
        # return observation, total_reward, done, info

    def reset(self, seed=None, options=None):
        if self.pyboy_counter == 10000:
            self.pyboy.stop()
            del self.pyboy
            self.pyboy_counter = 0
            self.pyboy = PyBoy('ROMs/Pokemon_Yellow.gbc', window_type="headless",
                               window_scale=3, game_wrapper=False)
        if seed:
            np.random.seed(seed)
        self.reset_game_state()
        self.reset_game_in_gui()
        if self.initial_observation:
             l_obs = self.get_image_observation()
             self.frame_stack.extend([l_obs] * self.frame_stack.maxlen)
        self.pyboy_counter += 1
        return self.get_observation(), {}

    def render(self):
        raw_screen = self.pyboy.botsupport_manager().screen().screen_ndarray()
        raw = np.array(raw_screen)[:, :, :3].astype(np.uint8)
        # gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        # resized = cv2.resize(edges, (60, 60))
        # resized = cv2.resize(edges, (120, 120))
        # rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Game', resized)  # Display the cropped image
        label = str(self.agent_id)
        cv2.imshow(label, raw)  # Display the cropped image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def close(self):
        cv2.destroyAllWindows()
        self.pyboy.stop()

    def get_image_observation(self):
        raw_screen = self.pyboy.botsupport_manager().screen().screen_ndarray()
        raw = np.array(raw_screen)[:, :, :3].astype(np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (120, 120))
        return resized[:, :, np.newaxis]

    def get_additional_features(self):
        x_coord = self.pyboy.get_memory_value(0xd35d)  # Replace with actual memory address for x-coordinate
        y_coord = self.pyboy.get_memory_value(0xd360)  # Replace with actual memory address for y-coordinate
        map_id = self.pyboy.get_memory_value(0xd361)  # Replace with actual memory address for map ID
        return np.array([x_coord, y_coord, map_id], dtype=np.float32)

    def get_observation(self):
        image_obs = self.get_image_observation()
        additional_features = self.get_additional_features()
        return {
            'image': image_obs,
            'additional_features': additional_features
        }

    def get_stacked_observation(self):
        # The shape of each frame should be (80, 72, 1)
        # After stacking along the last axis, the shape should be (80, 72, 4)
        return np.concatenate(self.frame_stack, axis=-1)

    def execute_action(self, action):
        action_map = {'0': 'a', '1': 'right', '2': 'left', '3': 'up', '4': 'down', '5': 'b'}
        if action == 5:
            self.release_all_keys()
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_B)
            for _ in range(self.wait_frames):  # tick for wait frames
                self.pyboy.tick()
        if action == 1:
            self.release_all_keys()
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            for _ in range(self.wait_frames):  # tick for wait frames
                self.pyboy.tick()
        if action == 2:
            self.release_all_keys()
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
            for _ in range(self.wait_frames):  # tick for wait frames
                self.pyboy.tick()
        if action == 3:
            self.release_all_keys()
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_UP)
            for _ in range(self.wait_frames):  # tick for wait frames
                self.pyboy.tick()
        if action == 4:
            self.release_all_keys()
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            for _ in range(self.wait_frames):  # tick for wait frames
                self.pyboy.tick()
        if action == 0:
            self.release_all_keys()
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            for _ in range(self.wait_frames):  # tick for wait frames
                self.pyboy.tick()

    def update_frame_stack(self):
        observation = self.get_image_observation()
        self.frame_stack.append(observation)

    def calculate_reward_and_done(self, action):
        reward = self.calculate_reward(action)
        self.total_reward += reward
        if self.truncated:
            done = True
        else:
            done = False
        if done:
            print(self.agent_id)
            print('episode_length = ', self.episode_length)
            # print('current_step = ', self.current_step)
            print('total_reward = ', self.total_reward)
            # print('Score recorded = ', self.current_score)
        return reward, done

    def calculate_reward(self, action):
        # Set reward variable
        reward = 0
        # Call get_score to see if a points sprite is detected in the obs
        current_score = self.get_score()
        # current_score = 0
        # If there is no score detected then it will return 0
        if current_score > 0:
            # If there is a score detected then it will return the reward that matches
            reward += current_score  # Setting reward equal to the score difference
            print(self.agent_id)
            print("Total Lvls went up by=", current_score)
        # encourage battling by rewarding when in battle mode
        if self.battling():
            reward += 1
            print(self.agent_id)
            print("got battling reward")
        if self.did_damage():
            if self.new_enemy_hp == 0:
                reward += 10
                print(self.agent_id)
                print("beat pokemon reward")
            else:
                reward += 0.01
                print(self.agent_id)
                print("did damage reward")
        # Check if the total HP of your pokemon has dropped and penalise
        if self.did_hp_drop():
            reward += -0.01
            print(self.agent_id)
            print("HP dropped")
        # print("No new points scored, reward remains 0")
        was_flag_reached = self.found_flag()
        if was_flag_reached:
            reward += 2000
            # print('flag reached')
        is_episode_finished = self.timed_out()
        if is_episode_finished:
            # print('Timed Out at episode', self.episode_length)
            reward += 0
            self.truncated = True
        # Call the function to detect if Mario is alive
        did_ash_faint = self.detect_ash_faint()
        # print("did_ash_faint = ",did_ash_faint)
        if did_ash_faint:
            # Apply the penalty for falling and set penalty_cap to True to avoid repeated penalties
            reward += -1  # Assigning a negative reward
            self.truncated = True
            print("Ash is fainted")
        did_ash_get_stuck = self.is_ash_stuck()
        if did_ash_get_stuck:
            if self.ash_stuck_counter >= 3000:
                reward += -0.01
                # print("Ash is stuck")
                # print("Ash stuck Counter =", self.ash_stuck_counter)
                if self.ash_stuck_counter >= 5000:
                    self.truncated = True
                    reward += -1
                    print(self.agent_id)
                    print("Ash not unstuck, exiting")
            else:
                if self.is_battling_fl != True:
                    reward += 0.001
                    # print(self.agent_id)
                    # print("Ash got explore reward")
        did_level_progress = self.level_did_progress()
        if did_level_progress > 0:
            # mu = 0.80  # Shifts the peak towards the end
            # sigma = 0.2  # Adjusts how quickly the rewards increase/decrease
            # gaussian_reward = np.exp(-np.power(did_level_progress - mu, 2.) / (2 * np.power(sigma, 2.)))
            # reward += gaussian_reward * 1000  # Scale the reward
            print(self.agent_id)
            print("level progressed = ", self.level_progress)
            reward += 2

        return reward

    def detect_ash_faint(self):
        pass
        return False

    def did_hp_drop(self):
        player_hp_addresses = [0xD16B, 0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C]
        # player_max_hp_addresses = [0xD18C, 0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D]
        total_hp = sum([self.pyboy.get_memory_value(address) for address in player_hp_addresses])
        # total_max_hp = sum([self.pyboy.get_memory_value(address) for address in player_max_hp_addresses])
        hp_drop = False
        if self.new_total_hp == 0:
            self.new_total_hp = total_hp
            return hp_drop
        # print("Total Pokemon HP = ",total_hp)
        if total_hp < self.new_total_hp:
            self.new_total_hp = total_hp
            hp_drop = True
        return hp_drop

    def did_damage(self):
        damaged = False
        if self.is_battling_fl:
            enemy_hp = self.pyboy.get_memory_value(0xcfe6)
            if enemy_hp < self.new_enemy_hp:
                damaged = True
                self.new_enemy_hp = enemy_hp
                return damaged
        else:
            return damaged

    def reset_game_in_gui(self):
        for _ in range(self.wait_frames):  # tick for wait frames
            self.pyboy.tick()
        # List of specific filenames
        gamestate_filenames = [
            "ROMs/Pokemon_Yellow.gbc.state.old2"
        ]
        # Select a random filename from the list
        selected_filename = random.choice(gamestate_filenames)
        file_like_object = open(selected_filename, "rb")
        self.pyboy.load_state(file_like_object)

    def get_score(self):
        d_1 = self.pyboy.get_memory_value(0xd18b)
        # print(d_1)
        d_2 = self.pyboy.get_memory_value(0xd1b7)
        # print(d_2)
        d_3 = self.pyboy.get_memory_value(0xd1e3)
        # print(d_3)
        d_4 = self.pyboy.get_memory_value(0xd20f)
        # print(d_4)
        d_5 = self.pyboy.get_memory_value(0xd23b)
        # print(d_5)
        d_6 = self.pyboy.get_memory_value(0xd267)
        # print(d_6)
        values = d_1 + d_2 + d_3 + d_4 + d_5 + d_6
        # print("score mem add values =",values)
        score = 0

        if values > self.current_score:
            if self.current_score == 0:
                self.current_score += values
                score = 0
            else:
                score = values - self.current_score
                score = score * 1
                self.current_score += score
        else:
            score = 0
        # print("score based reward =", score)
        return score

    def found_flag(self):
        found = False
        return found

    def battling(self):
        is_battling = False
        if self.pyboy.get_memory_value(0xd056) != 0:
            if self.is_battling_fl:
                return is_battling
            else:
                if self.new_enemy_hp != 0:
                    is_battling = False
                    self.is_battling_fl = True
                    self.new_enemy_hp = self.pyboy.get_memory_value(0xcfe6)
                    return is_battling
                else:
                    is_battling = True
                    self.is_battling_fl = True
                    self.new_enemy_hp = self.pyboy.get_memory_value(0xcfe6)
                    self.new_enemy_hp = self.pyboy.get_memory_value(0xcfe6)
                    self.new_enemy_hp = self.pyboy.get_memory_value(0xcfe6)
                    print("self.new_enemy_hp = ", self.pyboy.get_memory_value(0xcfe6))
                    return is_battling
        else:
            self.is_battling_fl = False
            # self.new_enemy_hp = 0
            return is_battling

    def timed_out(self):
        finish = False
        if self.episode_length > self.max_episodes:
            finish = True
        return finish

    def is_ash_stuck(self):
        stuck = False
        v_0 = self.pyboy.get_memory_value(0xd35d)
        v_1 = self.pyboy.get_memory_value(0xd360)
        v_2 = self.pyboy.get_memory_value(0xd361)
        v_3 = self.pyboy.get_memory_value(0xcc29)
        loc = (v_1 * v_2) * v_0
        last_20 = self.ash_loc_dict[-30:]
        if self.ash_is_moving == 0:
            self.ash_is_moving = loc
            return stuck
        if self.ash_is_moving == loc:
            if self.pyboy.get_memory_value(0xd056) == 0:
                stuck = True
                self.ash_stuck_counter += 1
                return stuck
        if loc in last_20:
            if self.pyboy.get_memory_value(0xd056) == 0:
                stuck = True
                self.ash_stuck_counter += 1
                return stuck
        if self.is_battling_fl:
            if v_3 == 3:
                self.battle_stuck_counter += 1
                if self.battle_stuck_counter > 1000:
                    stuck = True
                    self.ash_stuck_counter += 1
                    # nprint("battle is stuck count=",self.battle_stuck_counter)
                    return stuck
                else:
                    return stuck
        self.ash_loc_dict.append(loc)
        if len(self.ash_loc_dict) > 5000:
            self.ash_loc_dict.pop(0)
        self.ash_stuck_counter = 0
        self.battle_stuck_counter = 0
        self.ash_is_moving = loc
        return stuck

    def level_did_progress(self):
        v_1 = self.pyboy.get_memory_value(0xd35d)
        # check if map location is in the registered list
        if v_1 in self.level_progress:
            return 0
        else:
            if self.first_loc_check == 0:
                self.level_progress.append(v_1)
                self.first_loc_check = 1
                return 0
            else:
                self.level_progress.append(v_1)
                return 1

    def release_all_keys(self):
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_UP)
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
