from mss import mss
import cv2
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
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
    def __init__(self, max_episodes=500000):
        super().__init__()
        self.frame_stack = deque(maxlen=1)
        self.frame_skip = 1  # Number of frames to skip
        # Adjust observation space to 3D for CNN compatibility
        self.observation_space = Box(low=0, high=255, shape=(120, 120, 3), dtype=np.uint8)
        self.action_space = Discrete(6)
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
        self.ash_loc_dict = [(-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1),
                             (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1),
                             (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1),
                             (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1)]
        self.pokemon_found_list = [-1]
        self.pokemon_caught_list = [-1]
        self.level_progress_pct = 0.0
        self.ash_is_moving = -1
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
        self.first_episode = True
        self.previous_enemy_lvl = 1000
        self.map_steps_dict = {}
        self.global_goal = ''
        self.pkm_fnd = -1
        self.pkm_cau = -1
        self.pokemon_tally = 0
        print('STARTED AGENT: ', self.agent_id)

    def reset_game_state(self):
        self.total_reward = 0
        self.episode_length = 0
        self.current_score = 0
        self.level_progress = [-1]
        self.pokemon_found_list = [-1]
        self.pokemon_caught_list = [-1]
        self.ash_loc_dict = [(-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1),
                             (-1, -1, -1), (-1, -1, -1),
                             (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1),
                             (-1, -1, -1),
                             (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1),
                             (-1, -1, -1), (-1, -1, -1),
                             (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1),
                             (-1, -1, -1)]
        self.level_progress_pct = 0.0
        self.ash_is_moving = -1
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
        self.previous_enemy_lvl = 1000
        self.map_steps_dict = {}
        self.global_goal = ''
        self.pkm_fnd = -1
        self.pkm_cau = -1
        self.pokemon_tally = 0

    def step(self, action):
        total_reward = 0
        done = False
        info = {}
        for _ in range(self.frame_skip):
            if not done:
                self.execute_action(action)
                self.get_goal()
                self.update_frame_stack()
                observation = self.get_stacked_observation()
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
        save_file_object = open("ROMs/save_state_agent_{}".format(self.agent_id), "wb")
        self.pyboy.save_state(save_file_object)
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
            self.get_goal()
            l_obs = self.get_observation()
            self.frame_stack.extend([l_obs] * self.frame_stack.maxlen)
        self.pyboy_counter += 1
        return self.get_stacked_observation(), {}

    def render(self):
        goal = self.global_goal
        # print('render = ', goal)
        raw_screen = self.pyboy.botsupport_manager().screen().screen_ndarray()
        raw = np.array(raw_screen)[:, :, :3].astype(np.uint8)
        resized = cv2.resize(raw, (120, 120))
        last = self.add_color_block(resized, goal)

        label = str(self.agent_id)
        cv2.imshow(label, last)  # Display the cropped image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def close(self):
        cv2.destroyAllWindows()
        self.pyboy.stop()

    def get_observation(self):
        self.update_map_steps(self.get_map_id(),1)
        self.pyboy.set_memory_value(0xd31e, 99)
        goal = self.global_goal
        raw_screen = self.pyboy.botsupport_manager().screen().screen_ndarray()
        # print('raw_screen = ', np.shape(raw_screen))
        raw = np.array(raw_screen)[:, :, :3].astype(np.uint8)
        # gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)

        resized = cv2.resize(raw, (120, 120))
        resized = self.add_color_block(resized, goal)
        # return resized[:, :, np.newaxis]
        return resized

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
        observation = self.get_observation()
        self.frame_stack.append(observation)  # Each frame should have shape (80, 72, 1)

    def calculate_reward_and_done(self, action):
        goal = self.global_goal
        reward = self.calculate_reward_basic(action, goal)
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
            reward += 0
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
            reward += -0
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
            if self.ash_stuck_counter >= 5000:
                reward += -0.01
                # print("Ash is stuck")
                # print("Ash stuck Counter =", self.ash_stuck_counter)
                if self.ash_stuck_counter >= 10000:
                    self.truncated = True
                    reward += -10
                    print(self.agent_id)
                    print("Ash not unstuck, exiting")
        else:
            if self.is_battling_fl != True:
                v_0 = self.pyboy.get_memory_value(0xd35d)
                v_1 = self.pyboy.get_memory_value(0xd360)
                v_2 = self.pyboy.get_memory_value(0xd361)
                loc = self.calculate_location_hash(v_0, v_1, v_2)
                # loc = (v_1 * v_2) * v_0
                if loc not in self.ash_loc_dict:
                    reward += 0.001
                    self.ash_loc_dict.append(loc)
                    if len(self.ash_loc_dict) > 5000:
                        self.ash_loc_dict.pop(0)
                    print(self.agent_id)
                    print("Ash got explore reward")
            else:
                if self.new_pokemon_found() == 1:
                    reward += 10
                    print(self.agent_id)
                    print("Ash found a new pokemon reward")

        did_level_progress = self.level_did_progress()
        if did_level_progress > 0:
            # mu = 0.80  # Shifts the peak towards the end
            # sigma = 0.2  # Adjusts how quickly the rewards increase/decrease
            # gaussian_reward = np.exp(-np.power(did_level_progress - mu, 2.) / (2 * np.power(sigma, 2.)))
            # reward += gaussian_reward * 1000  # Scale the reward
            print(self.agent_id)
            print("level progressed = ", self.level_progress)
            reward += 10
        return reward

    def calculate_reward_basic(self, action, goal):
        # Set reward variable
        reward = 0
        is_episode_finished = self.timed_out()
        if is_episode_finished:
            # print('Timed Out at episode', self.episode_length)
            reward += 0
            self.truncated = True
        self.battling()
        did_ash_get_stuck = self.is_ash_stuck()
        if did_ash_get_stuck:
            if self.ash_stuck_counter >= 3000:
                reward += -0.01
                # print("Ash is stuck")
                # print("Ash stuck Counter =", self.ash_stuck_counter)
                if self.ash_stuck_counter >= 5000:
                    self.truncated = True
                    reward += -10
                    print(self.agent_id)
                    print("goal = ", goal)
                    print("Ash not unstuck, exiting")
        else:
            self.ash_stuck_counter = 0
            if not self.chk_battling():
                v_0 = self.pyboy.get_memory_value(0xd35d)
                v_1 = self.pyboy.get_memory_value(0xd360)
                v_2 = self.pyboy.get_memory_value(0xd361)
                loc = (v_1, v_2, v_0)
                self.ash_is_moving = loc
                if loc not in self.ash_loc_dict:
                    if goal == 'green':
                        reward += 0.001
                        print(self.agent_id)
                        print("goal = ", goal)
                        print("Ash got explore reward")
                    self.ash_loc_dict.append(loc)
                    if len(self.ash_loc_dict) > 5000:
                        self.ash_loc_dict.pop(0)
        if self.chk_battling():
            pkm_f = self.pkm_fnd
            if pkm_f not in self.pokemon_found_list:
                self.pokemon_found_list.append(pkm_f)
                if goal == 'magenta':
                    reward += 1
                    print(self.agent_id)
                    print("goal = ", goal)
                    print("Ash discovered a Pokemon")
                    # self.pokemon_found_list.append(self.pyboy.get_memory_value(0xcfd9))
            pkm_c = self.pkm_cau
            if pkm_c not in self.pokemon_caught_list:
                v_1 = self.pokemon_caught()
                if v_1 > self.pokemon_tally:
                    self.pokemon_caught_list.append(pkm_c)
                    self.pokemon_tally = v_1
                    if goal == 'magenta':
                        reward += 10
                        print(self.agent_id)
                        print("goal = ", goal)
                        print("Ash caught a Pokemon")
                        # self.pokemon_caught_list.append(self.pyboy.get_memory_value(0xcfd9))
            current_score = self.get_score()
            if current_score >= 10:
                # If there is a score detected then it will return the reward that matches
                if goal == 'red':
                    reward += current_score  # Setting reward equal to the score difference
                    print(self.agent_id)
                    print("goal = ", goal)
                    print("reward: Total Lvls went up by=", (current_score/10))
            # encourage battling by rewarding when in battle mode
        if self.battling():
            if goal == 'red':
                reward += 0.1
                print(self.agent_id)
                print("goal = ", goal)
                print("battling")
        if self.did_damage():
            if self.new_enemy_hp == 0:
                if goal == 'red':
                    reward += 1
                    print(self.agent_id)
                    print("goal = ", goal)
                    print("beat pokemon")
                else:
                    reward += 0
                    print(self.agent_id)
                    print("goal = ", goal)
                    print("did damage")
        # Check if the total HP of your pokemon has dropped and penalise
        if self.did_hp_drop():
            if goal == 'red':
                reward += 0
                print(self.agent_id)
                print("goal = ", goal)
                print("HP dropped")
        # print("No new points scored, reward remains 0")
        did_level_progress = self.level_did_progress()
        if did_level_progress > 0:
            if goal == 'blue':
                print(self.agent_id)
                print("goal = ", goal)
                print("level progressed = ", self.level_progress)
                reward += 10
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
        if self.first_episode:
            selected_filename = "ROMs/Pokemon_Yellow.gbc.state"
            self.first_episode = False
        else:
            gamestate_filenames = [
                "ROMs/Pokemon_Yellow.gbc.state"
            ]
            # Select a random filename from the list
            selected_filename = random.choice(gamestate_filenames)
        file_like_object = open(selected_filename.format(self.agent_id), "rb")
        self.pyboy.load_state(file_like_object)

    def pokemon_caught(self):
        v_1 = self.pyboy.get_memory_value(0xd162)
        v_2 = self.pyboy.get_memory_value(0xda7f)
        values = v_1 + v_2
        # print("Total Pokemon Count = ", values)
        return values

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
                score = score * 10
                self.current_score += values
        else:
            score = 0
        # print("score based reward =", score)
        return score

    def found_flag(self):
        found = False
        return found

    def chk_battling(self):
        is_battling = False
        if self.pyboy.get_memory_value(0xd056) != 0:
            is_battling = True
        return is_battling


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
                    self.previous_enemy_lvl = self.pyboy.get_memory_value(0xcff2)
                    return is_battling
                else:
                    is_battling = True
                    self.is_battling_fl = True
                    self.new_enemy_hp = self.pyboy.get_memory_value(0xcfe6)
                    self.previous_enemy_lvl = self.pyboy.get_memory_value(0xcff2)
                    # print("self.new_enemy_hp = ", self.pyboy.get_memory_value(0xcfe6))
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
        loc = (v_1, v_2, v_0)
        last_20 = self.ash_loc_dict[-30:]
        if self.ash_is_moving == (-1, -1, -1):
            # print("First check ash_is_moving")
            return stuck
        self.ash_is_moving = loc
        if loc in last_20:
            if self.pyboy.get_memory_value(0xd056) == 0:
                stuck = True
                self.ash_stuck_counter += 1
                # print("Ash is still in last 30 location zone=",self.ash_stuck_counter)
                return stuck
            else:
                self.ash_stuck_counter = 0
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

    def calculate_location_hash(self, map_id, x_coord, y_coord):
        # Ensure the coordinates are within 0-255 range
        x_coord = x_coord % 256
        y_coord = y_coord % 256

        # Create a unique hash
        # This method allows for 256 maps, each with 256x256 coordinates
        return (map_id << 16) | (x_coord << 8) | y_coord

    def new_pokemon_found(self):
        if self.chk_battling():
            v_1 = self.pyboy.get_memory_value(0xcfd9)
            if v_1 not in self.pokemon_found_list:
                self.pkm_fnd = v_1
                # print(v_1)
                # print(self.pokemon_found_list)
                # print("new pokemon discovered")
                return 1
            else:
                return 0
        else:
            return 0

    def new_pokemon_caught(self):
        if self.chk_battling():
            v_1 = self.pyboy.get_memory_value(0xcfd9)
            if v_1 not in self.pokemon_caught_list:
                self.pkm_cau = v_1
                # print (v_1)
                # print(self.pokemon_caught_list)
                # print("pokemon encountered but not caught")
                return 1
            else:
                return 0
        else:
            return 0

    def get_goal(self):
        total_lvls = self.current_score
        enemy_lvl = self.previous_enemy_lvl
        map_steps = self.get_map_steps(self.get_map_id())
        new_pkm_fnd = self.new_pokemon_found()
        new_pkm_cau = self.new_pokemon_caught()
        num_poke_bll = self.pyboy.get_memory_value(0xd31e)


        # Goal decision tree
        # First Check total Pokemon Strength
        if total_lvls > (enemy_lvl * 10):
            too_strong = True
        else:
            too_strong = False
        # Next Check total map steps taken
        # if map_steps > 10000000000000:
        #    too_many_steps = True
        # else:
        #    too_many_steps = False
        if self.ash_stuck_counter > 3000:
            too_many_steps = True
        else:
            too_many_steps = False
        # finally check number of pokeballs in bag
        if num_poke_bll == 0:
            no_poke_bll = True
        else:
            no_poke_bll = False
        # Goal setting
        # Decision logic
        if self.pyboy.get_memory_value(0xd056) == 1:
            # print('evaluated as battling wild pokemon')
            # print('new_pkm_fnd = ', new_pkm_fnd)
            if new_pkm_fnd == 1:
                # print("new pokemon goal set")
                goal = 'magenta' # Catch Pokemon
                self.global_goal = goal
                return goal
            if  new_pkm_cau == 1:
                # print('caught pokemon goal set')
                goal = 'magenta' # Catch Pokemon
                self.global_goal = goal
                return goal
        if too_many_steps:
            goal = 'green' # explore location
            self.global_goal = goal
            return goal
        if too_strong is False:
            goal = 'red' # Power up Pokemon
            self.global_goal = goal
            return goal
        if too_strong:
            goal = 'green' # Explore Location
            self.global_goal = goal
            return goal
        # elif no_poke_bll:
            # goal = 'magenta' # Find poke mart and buy balls
        goal = 'green' # default to map seeking
        self.global_goal = goal
        return goal

    # Function to update the dictionary
    def update_map_steps(self, map_id, steps_taken):
        if map_id in self.map_steps_dict:
            self.map_steps_dict[map_id] += steps_taken
        else:
            self.map_steps_dict[map_id] = steps_taken

    def get_map_steps(self,map_id):
        return self.map_steps_dict.get(map_id, 0)

    def get_map_id(self):
        map_id = self.pyboy.get_memory_value(0xd35d)
        return map_id

    def add_color_block(self, image, color_name):
        color_map = {
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'magenta': (255, 0, 255)
            # Add more colors as needed
        }
        # Convert color name to RGB value
        if color_name not in color_map:
            raise ValueError(
                f"Color '{color_name}' not recognized. Please use one of the following colors: {', '.join(color_map.keys())}")
        color = color_map[color_name]
        # Create a 20x20 pixel block of the specified color
        color_block = np.full((20, 20, 3), color, dtype=np.uint8)
        # Overlay the color block on the top left corner of the image
        image[0:20, 0:20] = color_block
        return image

    def release_all_keys(self):
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_UP)
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
