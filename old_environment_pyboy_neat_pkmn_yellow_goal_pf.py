from mss import mss
import time
import heapq
import cv2
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
import os
from collections import deque, defaultdict
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import random
from typing import List, Tuple, Optional
from PIL import Image  # Ensure PIL is installed
import logging
import matplotlib.pyplot as plt

# --------------------- Logging Configuration ---------------------
# Configure logging
logging.basicConfig(
    level=logging.CRITICAL,  # Set the logging level to CRITICAL to reduce verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log message format
    handlers=[
        logging.StreamHandler()  # Output logs to the console
    ]
)

# Create a logger for this module
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------


class GbaGame(Env):
    def __init__(self, max_episodes=500000):
        super().__init__()
        # Initialize necessary attributes
        self.curiosity_map = defaultdict(lambda: defaultdict(lambda: 1.0))  # High curiosity initially
        self.curiosity_decay_rate = 0.1  # Define how fast curiosity wanes with visits
        self.frame_stack = deque(maxlen=1)
        self.frame_skip = 1  # Number of frames to skip
        self.observation_space = Box(low=0, high=255, shape=(120, 120, 3), dtype=np.uint8)
        self.action_space = Discrete(6)
        self.pyboy = PyBoy('ROMs/Pokemon_Yellow.gbc', window_type="headless", scale=3)
        self.collision_counts = defaultdict(int)
        self.map_size = 100
        self.map_offset = self.map_size // 2
        self.cell_size = 10
        self.map_image = np.ones(
            (self.map_size * self.cell_size, self.map_size * self.cell_size, 3), dtype=np.uint8
        ) * 255
        self.map = defaultdict(lambda: defaultdict(int))
        self.encounter_positions = defaultdict(set)
        # Modify transitions to store destination map IDs
        self.transitions = defaultdict(set)
        self.current_map_id = None
        self.ash_position = None
        self.ash_old_position = None
        self.frontiers = set()
        self.current_path = []
        self.current_frontier = None
        self.map_offsets = {}
        self.max_episodes = max_episodes
        self.agent_id = random.randint(1, 1000000)
        self.first_episode = True
        self.initial_observation = True
        self.pyboy_counter = 0
        print('STARTED AGENT: ', self.agent_id)
        self.reset_game_state()
        # Extras
        self.total_reward = 0
        self.episode_length = 0
        self.current_score = 0
        self.level_progress = [-1]
        self.pokemon_found_list = [-1]
        self.pokemon_caught_list = [-1]
        self.level_progress_pct = 0.0
        self.ash_stuck_counter = 0
        self.is_battling_fl = False
        self.truncated = False
        self.new_total_hp = 0
        self.first_loc_check = 0
        self.new_enemy_hp = 0
        self.previous_enemy_lvl = 1000
        self.global_goal = ''
        self.pkm_fnd = -1
        self.pkm_cau = -1
        self.pokemon_tally = 0
        self.last_action = None
        self.previous_action = None
        self.preferred_action = None
        self.ash_loc_history = deque(maxlen=50)
        self.visited_frontiers = set()
        self.steps_since_last_frontier = 0
        self.steps_at_same_position = 0
        self.random_action_steps_remaining = 0
        self.goal_count = 0
        self.rand_count = 0
        # Add these lines
        self.excluded_frontiers = set()
        self.steps_towards_current_frontier = 0
        # Add fully explored maps set
        self.fully_explored_maps = set()
        self.transition_positions = defaultdict(set)  # map_id -> set of (x, y)
        self.last_transition_index = -1
        self.frame_wait = 24
        self.frame_wait_count = 0
        self.ppo_action = 0
        self.old_frontier = None
        self.ash_hp = 0

    def reset_game_state(self):
        self.total_reward = 0
        self.episode_length = 0
        self.current_score = 0
        self.level_progress = [-1]
        self.pokemon_found_list = [-1]
        self.pokemon_caught_list = [-1]
        self.level_progress_pct = 0.0
        self.ash_stuck_counter = 0
        self.is_battling_fl = False
        self.truncated = False
        self.new_total_hp = 0
        self.first_loc_check = 0
        self.new_enemy_hp = 0
        self.previous_enemy_lvl = 1000
        self.global_goal = ''
        self.pkm_fnd = -1
        self.pkm_cau = -1
        self.pokemon_tally = 0
        self.last_action = None
        self.previous_action = None
        self.preferred_action = None
        self.ash_loc_history = deque(maxlen=50)
        self.visited_frontiers = set()
        self.steps_since_last_frontier = 0
        self.steps_at_same_position = 0
        self.random_action_steps_remaining = 0
        self.goal_count = 0
        self.rand_count = 0
        # Add these lines
        self.excluded_frontiers = set()
        self.steps_towards_current_frontier = 0
        self.map = defaultdict(lambda: defaultdict(int))
        self.encounter_positions = defaultdict(set)
        self.transitions = defaultdict(set)
        self.current_map_id = None
        self.ash_position = None
        self.ash_old_position = None
        self.frontiers = set()
        self.current_path = []
        self.current_frontier = None
        self.map_offsets = {}
        self.map_image = np.ones((self.map_size * self.cell_size, self.map_size * self.cell_size, 3), dtype=np.uint8) * 255
        # Add fully explored maps set
        self.fully_explored_maps = set()
        self.transition_positions = defaultdict(set)  # map_id -> set of (x, y)
        self.last_transition_index = -1
        self.frame_wait_count = 0
        self.old_frontier = None
        self.ash_hp = 0

    def step(self, action):
        reward = 0
        done = False
        info = {}
        for _ in range(self.frame_skip):
            if not done:
                self.get_goal()
                self.battling()
                self.ppo_action = action
                previous_ash_position = self.ash_position
                if self.frame_wait_count <= self.frame_wait:
                    self.last_action = None
                    self.execute_action(self.last_action)
                    self.pyboy.tick(1, False)
                    if not self.is_battling_fl:
                        self.ash_old_position = self.ash_position
                        self.ash_position = self.get_ash_position()
                        self.ash_loc_history.append(self.ash_position)
                        self.update_slam_map()

                    self.update_frame_stack()

                    self.episode_length += 1
                    reward, done = self.calculate_reward_and_done(self.last_action)
                    truncated = self.truncated
                    if truncated:
                        info = {}
                    self.frame_wait_count += 1
                    return self.get_stacked_observation(), reward, done, truncated, info
                self.frame_wait_count = 0
                if self.global_goal == 'blue' and not self.is_battling_fl:
                    action = self.slam_action()
                    self.last_action = action
                    self.execute_action(self.last_action)
                    self.pyboy.tick(1, False)
                    self.ash_old_position = self.ash_position
                    self.ash_position = self.get_ash_position()
                    self.ash_loc_history.append(self.ash_position)
                    self.update_slam_map()
                else:
                    self.last_action = action
                    self.execute_action(self.last_action)
                    self.pyboy.tick(1, False)
                    if not self.is_battling_fl:
                        self.ash_old_position = self.ash_position
                        self.ash_position = self.get_ash_position()
                        self.ash_loc_history.append(self.ash_position)
                        self.update_slam_map()

                if not self.is_battling_fl:
                    if self.ash_position == previous_ash_position:
                        self.steps_at_same_position += 1
                    else:
                        self.steps_at_same_position = 0
                else:
                    self.steps_at_same_position = 0

                self.update_frame_stack()
                self.render()
                self.render_map()
                self.episode_length += 1
                reward, done = self.calculate_reward_and_done(self.last_action)
                truncated = self.truncated
                if truncated:
                    info = {}

        return self.get_stacked_observation(), reward, done, truncated, info

    def reset(self, seed=None, options=None):
        if self.pyboy_counter == 10000:
            self.pyboy.stop()
            del self.pyboy
            self.pyboy_counter = 0
            self.pyboy = PyBoy('ROMs/Pokemon_Yellow.gbc', window_type="headless", scale=3, game_wrapper=False)
        if seed:
            np.random.seed(seed)
        self.reset_game_state()
        self.reset_game_in_gui()
        if self.initial_observation:
            self.get_goal()
            l_obs = self.get_observation()
            self.frame_stack.extend([l_obs] * self.frame_stack.maxlen)
        self.pyboy_counter += 1
        self.current_map_id = self.get_map_id()
        self.ash_position = self.get_ash_position()
        self.initialize_map_offset(self.current_map_id, self.ash_position[1], self.ash_position[2])
        self.add_to_map(self.ash_position, 1)
        self.detect_frontiers()
        return self.get_stacked_observation(), {}

    def initialize_map_offset(self, map_id, x, y):
        if map_id not in self.map_offsets:
            offset_x = self.map_size // 2 - x
            offset_y = self.map_size // 2 - y
            self.map_offsets[map_id] = (offset_x, offset_y)
            logger.debug(f"Initialized map offset for Map ID {map_id}: Offset X = {offset_x}, Offset Y = {offset_y}")

    def render(self):
        goal = self.global_goal
        raw_screen = self.pyboy.screen.ndarray
        raw_screen = cv2.cvtColor(raw_screen, cv2.COLOR_BGR2RGB)
        if raw_screen is None:
            print("No screen data available from PyBoy.")
            return
        raw = np.array(raw_screen)[:, :, :3].astype(np.uint8)
        resized = cv2.resize(raw, (360, 360))
        last = self.add_color_block(resized, goal)
        label = str(self.agent_id)
        cv2.imshow(label, last)
        self.render_map()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def render_map(self):
        if not self.ash_position:
            logger.debug("Ash position is not set. Skipping map rendering.")
            return

        map_id, ash_x_game, ash_y_game = self.ash_position
        offset_x, offset_y = self.map_offsets.get(map_id, (self.map_size // 2, self.map_size // 2))

        # Apply offsets only to Ash's position
        ash_x = ash_x_game + offset_x
        ash_y = ash_y_game + offset_y

        # Log current map IDs for debugging
        logger.debug(f"Rendering Map ID: {map_id}, Current Map ID: {self.current_map_id}")

        # Validate Ash's position within bounds
        if not (0 <= ash_x < self.map_size and 0 <= ash_y < self.map_size):
            logger.warning(f"Ash's position ({ash_x}, {ash_y}) is out of bounds on map {map_id}.")
            return

        # Initialize the map image with a white background
        self.map_image[:] = 255

        # Draw map cells (free spaces and obstacles)
        for (x, y), value in self.map[map_id].items():
            top_left = (x * self.cell_size, y * self.cell_size)
            bottom_right = ((x + 1) * self.cell_size, (y + 1) * self.cell_size)
            if value == 1:
                color = (0, 0, 0)  # Black for free space
            elif value == -1:
                color = (0, 255, 0)  # Green for obstacles
            else:
                color = None  # Unmarked or other types

            if color:
                cv2.rectangle(self.map_image, top_left, bottom_right, color, -1)

        # Draw encounter positions
        for (x, y) in self.encounter_positions[map_id]:
            top_left = (x * self.cell_size, y * self.cell_size)
            bottom_right = ((x + 1) * self.cell_size, (y + 1) * self.cell_size)
            cv2.rectangle(self.map_image, top_left, bottom_right, (255, 255, 0), -1)  # Yellow

        # Draw transitions
        brown_color = (42, 42, 165)
        for (x, y) in self.transitions[map_id]:
            top_left = (x * self.cell_size, y * self.cell_size)
            bottom_right = ((x + 1) * self.cell_size, (y + 1) * self.cell_size)
            cv2.rectangle(self.map_image, top_left, bottom_right, brown_color, -1)

        # Draw frontiers
        for frontier in list(self.frontiers):
            f_map_id, fx, fy = frontier
            if f_map_id != map_id:
                continue
            top_left = (fx * self.cell_size, fy * self.cell_size)
            bottom_right = ((fx + 1) * self.cell_size, (fy + 1) * self.cell_size)
            cv2.rectangle(self.map_image, top_left, bottom_right, (255, 0, 0), -1)  # Blue

        # Draw current frontier
        if self.current_frontier:
            f_map_id, fx, fy = self.current_frontier
            if f_map_id == map_id:
                top_left = (fx * self.cell_size, fy * self.cell_size)
                bottom_right = ((fx + 1) * self.cell_size, (fy + 1) * self.cell_size)
                cv2.rectangle(self.map_image, top_left, bottom_right, (0, 165, 255), -1)  # Orange

        # **Overlay the planned path in purple**
        if self.current_path and self.current_map_id == map_id:
            logger.debug("Rendering planned path on the map.")
            path_points = []
            for pos in self.current_path:
                _, path_x, path_y = pos
                # **Do not apply offsets to path_x and path_y**
                grid_x = path_x
                grid_y = path_y
                logger.debug(f"Path Point: ({grid_x}, {grid_y})")

                if 0 <= grid_x < self.map_size and 0 <= grid_y < self.map_size:
                    center = (grid_x * self.cell_size + self.cell_size // 2,
                              grid_y * self.cell_size + self.cell_size // 2)
                    path_points.append(center)

                    # Draw semi-transparent purple rectangles
                    top_left_rect = (grid_x * self.cell_size, grid_y * self.cell_size)
                    bottom_right_rect = ((grid_x + 1) * self.cell_size, (grid_y + 1) * self.cell_size)
                    overlay = self.map_image.copy()
                    cv2.rectangle(overlay, top_left_rect, bottom_right_rect, (128, 0, 128), -1)  # Purple fill
                    alpha = 0.3  # Transparency factor
                    cv2.addWeighted(overlay, alpha, self.map_image, 1 - alpha, 0, self.map_image)

                    # Draw circles at each path point
                    cv2.circle(self.map_image, center, radius=2, color=(128, 0, 128), thickness=-1)
                    logger.debug(f"Drew path point at ({grid_x}, {grid_y})")

            # Draw lines connecting the path points
            if len(path_points) >= 2:
                for i in range(1, len(path_points)):
                    cv2.line(self.map_image, path_points[i - 1], path_points[i], color=(128, 0, 128), thickness=1)
                    logger.debug(f"Drew line from {path_points[i - 1]} to {path_points[i]}")

        # **Draw Ash's position**
        top_left = (ash_x * self.cell_size, ash_y * self.cell_size)
        bottom_right = ((ash_x + 1) * self.cell_size, (ash_y + 1) * self.cell_size)
        cv2.rectangle(self.map_image, top_left, bottom_right, (0, 0, 255), -1)  # Red fill
        cv2.rectangle(self.map_image, top_left, bottom_right, (255, 0, 0), 1)  # Blue border

        # **Display the map with the path**
        cv2.imshow("SLAM Map", self.map_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def close(self):
        cv2.destroyAllWindows()
        self.pyboy.stop()

    def get_observation(self):
        self.pyboy.memory[0xd31e] = 99
        goal = self.global_goal
        raw_screen = self.pyboy.screen.ndarray
        raw_screen = cv2.cvtColor(raw_screen, cv2.COLOR_BGR2RGB)
        if raw_screen is None:
            print("No screen data available from PyBoy.")
            return np.zeros((120, 120, 3), dtype=np.uint8)
        raw = np.array(raw_screen)[:, :, :3].astype(np.uint8)
        resized = cv2.resize(raw, (120, 120))
        resized = self.add_color_block(resized, goal)
        return resized

    def get_stacked_observation(self):
        return np.concatenate(self.frame_stack, axis=-1)

    def execute_action(self, action):
        self.release_all_keys()
        key_event = {
            1: WindowEvent.PRESS_ARROW_RIGHT,
            2: WindowEvent.PRESS_ARROW_LEFT,
            3: WindowEvent.PRESS_ARROW_UP,
            4: WindowEvent.PRESS_ARROW_DOWN,
            0: WindowEvent.PRESS_BUTTON_A,
            5: WindowEvent.PRESS_BUTTON_B,
        }.get(action)
        if key_event:
            ticks_needed = 1
            if action in [1, 2, 3, 4] and not self.is_battling_fl:
                if self.previous_action is not None:
                    opposite_actions = {1: 2, 2: 1, 3: 4, 4: 3}
                    if action == opposite_actions.get(self.previous_action):
                        ticks_needed = 2
            self.pyboy.send_input(key_event)
            logger.debug(f"Executing action {action}: Sent {key_event}")
            for _ in range(ticks_needed):
                self.pyboy.tick(1, True)
            if action in [1, 2, 3, 4]:
                self.previous_action = action
                logger.debug(f"Updated previous_action to {self.previous_action}")

    def update_frame_stack(self):
        observation = self.get_observation()
        self.frame_stack.append(observation)

    def calculate_reward_and_done(self, action):
        reward = self.calculate_reward_basic(action)
        self.total_reward += reward
        if self.truncated:
            done = True
        else:
            done = False
        if done:
            print(f"Agent ID: {self.agent_id}")
            print('Episode Length = ', self.episode_length)
            print('Total Reward = ', self.total_reward)
        return reward, done

    def calculate_reward_basic(self, action):
        reward = 0
        if self.timed_out():
            self.truncated = True
            print("Episode timed out.")
        loc = self.ash_position[1:]
        map_id = self.ash_position[0]
        offset_x, offset_y = self.map_offsets.get(map_id, (self.map_offset, self.map_offset))
        map_x = loc[0] + offset_x
        map_y = loc[1] + offset_y
        if self.map[self.ash_position[0]].get((map_x, map_y), 0) != 1:
            if self.global_goal == 'blue':
                reward += 0.001
            self.map[self.ash_position[0]][(map_x, map_y)] = 1
        if self.is_ash_stuck():
            if self.ash_stuck_counter >= 3000:
                reward -= 0.01
                if self.ash_stuck_counter >= 5000:
                    reward -= 10
        else:
            self.ash_stuck_counter = 0
        if self.chk_battling():
            pkm_f = self.pkm_fnd
            if pkm_f not in self.pokemon_found_list:
                self.pokemon_found_list.append(pkm_f)
                if self.global_goal == 'magenta':
                    reward += 1
            pkm_c = self.pkm_cau
            if pkm_c not in self.pokemon_caught_list:
                v_1 = self.pokemon_caught()
                if v_1 > self.pokemon_tally:
                    self.pokemon_caught_list.append(pkm_c)
                    self.pokemon_tally = v_1
                    if self.global_goal == 'magenta':
                        reward += 10
            current_score = self.get_score()
            if current_score == 10:
                if self.global_goal == 'red':
                    reward += current_score
        if self.battling():
            if not self.is_battling_fl:
                if self.global_goal == 'red':
                    reward += 0.1
        if self.did_damage():
            if self.new_enemy_hp == 0:
                if self.global_goal == 'red':
                    reward += 1
        if self.did_hp_drop():
            if self.global_goal == 'red':
                reward += 0
        if self.level_did_progress() > 0:
            if self.global_goal == 'blue':
                reward += 10
        return reward

    def did_hp_drop(self):
        player_hp_addresses = [0xD16B, 0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C]
        total_hp = sum([self.pyboy.memory[address] for address in player_hp_addresses])
        hp_drop = False
        if self.new_total_hp == 0:
            self.new_total_hp = total_hp
            return hp_drop
        if total_hp < self.new_total_hp:
            self.new_total_hp = total_hp
            hp_drop = True
        return hp_drop

    def did_damage(self):
        damaged = False
        if self.is_battling_fl:
            enemy_hp = self.pyboy.memory[0xcfe6]
            if enemy_hp < self.new_enemy_hp:
                damaged = True
                self.new_enemy_hp = enemy_hp
                return damaged
        return damaged

    def reset_game_in_gui(self):
        for _ in range(1):
            self.pyboy.tick()
        if self.first_episode:
            selected_filename = "ROMs/Pokemon_Yellow.gbc.state"
            self.first_episode = False
        else:
            selected_filename = "ROMs/Pokemon_Yellow.gbc.state"
        if not os.path.exists(selected_filename):
            raise FileNotFoundError(f"Save state file {selected_filename} does not exist.")
        with open(selected_filename, "rb") as file_like_object:
            self.pyboy.load_state(file_like_object)

    def pokemon_caught(self):
        v_1 = self.pyboy.memory[0xd162]
        v_2 = self.pyboy.memory[0xda7f]
        values = v_1 + v_2
        return values

    def get_score(self):
        addresses = [0xd18b, 0xd1b7, 0xd1e3, 0xd20f, 0xd23b, 0xd267]
        values = sum([self.pyboy.memory[addr] for addr in addresses])
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
        return score

    def chk_battling(self):
        player_hp_addresses = [0xD16B, 0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C]
        total_hp = sum([self.pyboy.memory[address] for address in player_hp_addresses])
        self.ash_hp = total_hp
        return self.pyboy.memory[0xd056] != 0

    def battling(self):
        battle_state = self.pyboy.memory[0xd056]
        if battle_state != 0:
            player_hp_addresses = [0xD16B, 0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C]
            total_hp = sum([self.pyboy.memory[address] for address in player_hp_addresses])
            self.ash_hp = total_hp
            if not self.is_battling_fl:
                self.is_battling_fl = True
                self.new_enemy_hp = self.pyboy.memory[0xcfe6]
                self.previous_enemy_lvl = self.pyboy.memory[0xcff2]
                current_grid_pos = self.get_ash_position_grid()
                self.encounter_positions[current_grid_pos[0]].add((current_grid_pos[1], current_grid_pos[2]))
                return True
        else:
            if self.is_battling_fl:
                self.is_battling_fl = False
                return False

    def timed_out(self):
        return self.episode_length > self.max_episodes

    def is_ash_stuck(self):
        if self.chk_battling():
            return False
        stuck = False
        loc = self.get_ash_position()
        self.ash_loc_history.append(loc)
        if len(self.ash_loc_history) == self.ash_loc_history.maxlen:
            if len(set(self.ash_loc_history)) <= 2:
                stuck = True
                self.ash_stuck_counter += 1
            else:
                self.ash_stuck_counter = 0
        return stuck

    def level_did_progress(self):
        v_1 = self.get_map_id()
        if v_1 in self.level_progress:
            return 0
        else:
            self.level_progress.append(v_1)
            return 1

    def new_pokemon_found(self):
        if self.chk_battling():
            v_1 = self.pyboy.memory[0xcfd9]
            if v_1 not in self.pokemon_found_list:
                self.pkm_fnd = v_1
                return 1
        return 0

    def new_pokemon_caught(self):
        if self.chk_battling():
            v_1 = self.pyboy.memory[0xcfd9]
            if v_1 not in self.pokemon_caught_list:
                self.pkm_cau = v_1
                return 1
        return 0

    def get_goal(self):
        total_lvls = self.current_score
        enemy_lvl = self.previous_enemy_lvl if self.previous_enemy_lvl > 0 else 1
        new_pkm_fnd = self.new_pokemon_found()
        new_pkm_cau = self.new_pokemon_caught()
        num_poke_balls = self.pyboy.memory[0xd31e]

        too_strong = total_lvls > (enemy_lvl * 15)
        too_many_steps = self.ash_stuck_counter > 3000
        no_poke_balls = num_poke_balls == 0

        battle_state = self.pyboy.memory[0xd056]
        player_control = self.pyboy.memory[0xc507]
        if battle_state == 2:
            goal = 'red'
        elif battle_state == 1:
            if new_pkm_fnd == 1 or new_pkm_cau == 1:
                goal = 'magenta'
            elif too_strong:
                goal = 'green'
            else:
                goal = 'red'
        else:
            if player_control == 126 and battle_state == 0:
                goal = 'red'
            elif too_many_steps:
                if self.goal_count <= 500000:
                    goal = 'blue'
                    self.goal_count += 1
                    logger.info(f"self.goal_count = {self.goal_count}")
                else:
                    if self.rand_count <= 1000:
                        goal = 'green'
                        self.rand_count += 1
                    else:
                        self.goal_count = 0
                        self.rand_count = 0
                        goal = 'green'
            elif not too_strong:
                if self.goal_count <= 500000:
                    goal = 'blue'
                    self.goal_count += 1
                    logger.info(f"self.goal_count = {self.goal_count}")
                else:
                    if self.rand_count <= 1000:
                        goal = 'red'
                        self.rand_count += 1
                    else:
                        self.goal_count = 0
                        self.rand_count = 0
                        goal = 'red'
            elif too_strong and battle_state == 0:
                if self.goal_count <= 500000:
                    goal = 'blue'
                    self.goal_count += 1
                    logger.info(f"self.goal_count = {self.goal_count}")
                else:
                    if self.rand_count <= 1000:
                        goal = 'green'
                        self.rand_count += 1
                    else:
                        self.goal_count = 0
                        self.rand_count = 0
                        goal = 'green'
            else:
                goal = 'green'

        self.global_goal = goal
        return goal

    def get_map_id(self):
        return self.pyboy.memory[0xd35d]

    def add_color_block(self, image, color_name):
        color_map = {
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'magenta': (255, 0, 255)
        }
        if color_name not in color_map:
            raise ValueError(
                f"Color '{color_name}' not recognized. Please use one of the following colors: {', '.join(color_map.keys())}")
        color = color_map[color_name]
        color_block = np.full((20, 20, 3), color, dtype=np.uint8)
        image[0:20, 0:20] = color_block
        return image

    def release_all_keys(self):
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_UP)
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_B)

    def get_ash_position(self):
        y_coord = self.pyboy.memory[0xd360]
        x_coord = self.pyboy.memory[0xd361]
        map_id = self.pyboy.memory[0xd35d]
        self.current_map_id = map_id  # Update current_map_id
        return (map_id, x_coord, y_coord)

    def add_to_map(self, position, value):
        map_id, x, y = position
        # Re-initialize the map offset if not already set or after a transition
        if map_id not in self.map_offsets:
            logger.warning(f"Map ID {map_id} not found in offsets. Initializing with Ash's current position.")
            self.initialize_map_offset(map_id, x, y)
        offset_x, offset_y = self.map_offsets[map_id]
        grid_x = x + offset_x
        grid_y = y + offset_y
        current_value = self.map[map_id].get((grid_x, grid_y), 0)

        if value == -1:
            # Overwrite any previous value to mark as obstacle
            self.map[map_id][(grid_x, grid_y)] = -1
            logger.debug(f"Added obstacle at grid position ({grid_x}, {grid_y}) on map {map_id}.")
        elif value == 1:
            if current_value != 1:
                self.map[map_id][(grid_x, grid_y)] = 1
                self.curiosity_map[map_id][(grid_x, grid_y)] = max(
                    0.0, self.curiosity_map[map_id][(grid_x, grid_y)] - self.curiosity_decay_rate
                )
                logger.debug(f"Marked grid position ({grid_x}, {grid_y}) on map {map_id} as free.")

    def update_slam_map(self):
        if len(self.ash_loc_history) < 50:
            return

        previous_position = self.ash_loc_history[-24]
        current_position = self.ash_position
        attempted_position = self.get_attempted_position(self.ash_old_position, self.last_action)

        if self.ash_old_position and self.ash_position:

            previous_map_id = self.ash_old_position[0]
            current_map_id = self.ash_position[0]
            if previous_map_id != current_map_id and self.ash_hp != 0:
                if self.current_frontier:
                    previous_map_id, prev_x, prev_y = self.current_frontier
                    logging.critical(f"update_slam_map self.current_frontier = {self.current_frontier}")
                elif self.old_frontier:
                    previous_map_id, prev_x, prev_y = self.old_frontier
                    logging.critical(f"update_slam_map self.old_frontier = {self.old_frontier}")
                else:
                    return
                prev_offset_x, prev_offset_y = self.map_offsets[previous_map_id]
                self.transitions[previous_map_id].add((prev_x, prev_y))
                self.handle_map_transition(previous_map_id, prev_x, prev_y)
                if self.current_frontier:
                    self.old_frontier = self.current_frontier
                self.current_frontier = None
                self.current_path = []

        if previous_position == current_position and self.frontiers:
            if self.last_action in [1, 2, 3, 4]:
                attempted_position = self.get_attempted_position(previous_position, self.last_action)
                collision = self.check_collision(attempted_position)
                if collision and self.global_goal == "blue":
                    self.increment_collision_count(attempted_position)
                    collision_count = self.get_collision_count(attempted_position)
                    logger.debug(f"Collision detected at {attempted_position}. Collision count: {collision_count}")
                    if collision_count >= 15:
                        self.add_to_map(attempted_position, -1)
                        logger.info(f"Marked position {attempted_position} as an obstacle due to repeated collisions.")
                        self.current_path = []
                        self.reset_collision_count(attempted_position)
        else:
            self.add_to_map(current_position, 1)
            logger.debug(f"Updated map with free position: {current_position}")

        self.detect_frontiers()
        

    def detect_frontiers(self):
        map_id, x, y = self.ash_position
        ash_grid_x, ash_grid_y = self.get_ash_position_grid()[1:]
        self.check_target_frontier_reached()
        possible_frontiers = []
        if not self.current_path:
            for (cell_x, cell_y), value in self.map[map_id].items():
                if value == 1 and not self.is_transition_position(map_id, cell_x, cell_y):
                    logger.debug(f"map_id = {map_id} , cell_x = {cell_x} , cell_y = {cell_y}")
                    logger.debug(f"self.transition_positions = {self.transition_positions}")
                    logger.debug(
                    f"self.is_transition_position(map_id, cell_x, cell_y) = {self.is_transition_position(map_id, cell_x, cell_y)}")
                    neighbors = self.get_neighbors((cell_x, cell_y))
                    for nx, ny in neighbors:
                        if self.map[map_id].get((nx, ny), 0) == 0:
                            if self.is_transition_position(map_id, nx, ny):
                                continue
                            distance = abs(nx - ash_grid_x) + abs(ny - ash_grid_y)
                            possible_frontiers.append((map_id, nx, ny, distance))
                            break

            if possible_frontiers:
                possible_frontiers.sort(key=lambda f: f[3])
                for frontier in possible_frontiers:
                    nearest_frontier = frontier[:3]
                    if (nearest_frontier not in self.visited_frontiers and
                        nearest_frontier not in self.excluded_frontiers):
                        self.frontiers = {nearest_frontier}
                        self.current_frontier = nearest_frontier
                        self.old_frontier = self.current_frontier
                        self.steps_since_last_frontier = 0
                        self.steps_towards_current_frontier = 0
                        logger.debug(f"New frontier detected: {nearest_frontier}")
                        break
            else:
                self.old_frontier = self.current_frontier
                self.current_frontier = None
                if not self.is_battling_fl:
                    self.steps_since_last_frontier += 1

    def is_transition_position(self, map_id: int, x: int, y: int) -> bool:
        return (x, y) in self.transition_positions.get(map_id, set())

    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = position
        neighbors = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                neighbors.append((nx, ny))
        return neighbors

    def check_target_frontier_reached(self):
        if self.current_frontier:
            ash_map_id, ash_x_grid, ash_y_grid = self.get_ash_position_grid()
            if (ash_map_id, ash_x_grid, ash_y_grid) == self.current_frontier:
                if self.is_transition_position(ash_map_id, ash_x_grid, ash_y_grid):
                    self.handle_map_transition(ash_map_id, ash_x_grid, ash_y_grid)
                else:
                    self.visited_frontiers.add(self.current_frontier)
                    logger.critical(f"Frontier {self.current_frontier} reached and marked as visited.")
                if self.current_frontier:
                    self.old_frontier = self.current_frontier
                self.current_frontier = None
                self.current_path = []
                self.steps_since_last_frontier = 0
                self.steps_towards_current_frontier = 0

    def handle_map_transition(self, map_id: int, x: int, y: int):
        logger.info(f"Map transition reached at ({map_id}, {x}, {y}). Handling transition.")
        self.transition_positions[map_id].add((x, y))
        self.excluded_frontiers.add((map_id, x, y))
        self.visited_frontiers.add((map_id, x, y))

    def slam_action(self) -> int:
        if self.current_path:
            next_position = self.current_path[0]
            current_position = self.get_ash_position_grid()

            if current_position == next_position:
                self.current_path.pop(0)
                logger.info(f"Ash moved to {next_position}. Progressing path.")

                if self.current_path:
                    next_position = self.current_path[0]
                else:
                    logger.critical("Ash has reached the final destination of the current path.")
                    return random.choice([1, 2, 3, 4])

            action = self.get_action_from_positions(current_position, next_position)

            logger.debug(f"Following path. Next action: {action} to move from {current_position} to {next_position}.")
            return action
        else:
            self.detect_frontiers()
            if self.current_frontier:
                logging.critical(f"current_frontier = {self.current_frontier}") 
                logging.critical(f"frontiers = {self.frontiers}")
                path = self.plan_path(self.get_ash_position_grid(), self.current_frontier)
                if path is None or len(path) < 2:
                    logger.critical("Cannot find a path to the frontier. Excluding the frontier.")
                    self.excluded_frontiers.add(self.current_frontier)
                    self.frontiers.discard(self.current_frontier)
                    if self.current_frontier:
                        self.old_frontier = self.current_frontier
                    self.current_frontier = None
                    self.goal_count = 500000
                    return self.ppo_action
                else:
                    self.current_path = path[1:]
                    logger.info(f"Planned path to frontier: {self.current_path}")
                    current_position = self.get_ash_position_grid()
                    next_position = self.current_path[0]
                    action = self.get_action_from_positions(current_position, next_position)
                    return action
            else:
                logger.critical(f"No current frontier, moving towards an unexplored map or fallback actions")
                moved = self.move_to_unexplored_map()
                if moved:
                    logger.critical("unexplored maps or transitions found. Returning path action.")
                    logger.info(f"Planned path to frontier: {self.current_path}")
                    current_position = self.get_ash_position_grid()
                    next_position = self.current_path[0]
                    action = self.get_action_from_positions(current_position, next_position)
                    return action
                known = self.move_towards_known_area()
                if known:
                    self.current_path = path[1:]
                    logger.info(f"Planned path to frontier: {self.current_path}")
                    current_position = self.get_ash_position_grid()
                    next_position = self.current_path[0]
                    action = self.get_action_from_positions(current_position, next_position)
                    logger.critical("known location selected. Returning path action.")
                    logger.critical(f"known = {known}")
                    return action
                    
                    return self.slam_action()
                else:
                    logger.critical("No unexplored maps or transitions left. Returning random action.")
                    if self.current_frontier:
                        self.old_frontier = self.current_frontier
                    self.current_frontier = None
                    self.current_path = []
                    self.steps_since_last_frontier = 0
                    self.steps_towards_current_frontier = 0
                    self.goal_count = 500000
                    return self.ppo_action

    def move_to_unexplored_map(self):
        current_map_id = self.current_map_id
        transitions = self.transitions.get(current_map_id, set())
        unexplored_transitions = []

        ash_map_id, ash_x_grid, ash_y_grid = self.get_ash_position_grid()
        if not self.current_path:
            for position in transitions:
                logging.critical(f"move to unexplored map triggered")
                cell_x, cell_y = position
                distance = abs(cell_x - ash_x_grid) + abs(cell_y - ash_y_grid)
                unexplored_transitions.append((position, current_map_id, distance))

            if unexplored_transitions:
                logging.critical(f"There are unexplored_transitions")
                unexplored_transitions.sort(key=lambda t: t[2], reverse=True)

                num_transitions = len(unexplored_transitions)
                logging.critical(f"num_transitions = {num_transitions}")

                if self.last_transition_index >= num_transitions:
                    self.last_transition_index = -1

                self.last_transition_index = (self.last_transition_index + 1) % num_transitions
                target_transition = unexplored_transitions[self.last_transition_index]
                target_position, dest_map_id, distance = target_transition

                logger.debug(f"Selected transition {self.last_transition_index + 1}/{num_transitions}: {target_transition}")

                self.frontiers = {(dest_map_id, target_position[0], target_position[1])}
                self.current_frontier = (dest_map_id, target_position[0], target_position[1])
                self.steps_since_last_frontier = 0
                self.steps_towards_current_frontier = 0
                path = self.plan_path(self.get_ash_position_grid(), self.current_frontier)

                if path is None or len(path) < 2:
                    logger.critical("Cannot find a path to the frontier. Excluding the frontier.")
                    return False
                else: 
                    self.current_path = path[1:]
                    logger.critical(f"Moving towards unexplored map. Path: {self.current_path}")
                    return True
        else:
            logger.critical("All neighboring maps are fully explored.")
            return False

    def move_towards_known_area(self):
        map_id = self.current_map_id
        ash_grid_x, ash_grid_y = self.get_ash_position_grid()[1:]
        known_positions = [(x, y) for (x, y), v in self.map[map_id].items() if v == 1]
        if not known_positions:
            return False
        known_positions.sort(key=lambda pos: abs(pos[0] - ash_grid_x) + abs(pos[1] - ash_grid_y))
        closest_known = known_positions[0]
        path = self.plan_path((map_id, ash_grid_x, ash_grid_y), (map_id, closest_known[0], closest_known[1]))
        if path and len(path) > 1:
            self.current_path = path[1:]
            logger.critical(f"Moving towards known area. Path: {self.current_path}")
            return True
        else:
            return False

    def plan_path(self, start, goal):
        start_map_id, start_x, start_y = start
        goal_map_id, goal_x, goal_y = goal
        if start_map_id != goal_map_id:
            return None

        if start == goal:
            return [start]

        open_set_fwd = []
        heapq.heappush(open_set_fwd, (0, (start_x, start_y)))
        came_from_fwd = {}
        g_score_fwd = defaultdict(lambda: float('inf'))
        g_score_fwd[(start_x, start_y)] = 0

        open_set_bwd = []
        heapq.heappush(open_set_bwd, (0, (goal_x, goal_y)))
        came_from_bwd = {}
        g_score_bwd = defaultdict(lambda: float('inf'))
        g_score_bwd[(goal_x, goal_y)] = 0

        closed_set_fwd = set()
        closed_set_bwd = set()
        max_search_area = 10000

        def heuristic(a, b):
            return (abs(a[0] - b[0]) + abs(a[1] - b[1])) * 0

        search_counter = 0
        meeting_node = None

        while open_set_fwd and open_set_bwd:
            if open_set_fwd:
                current_f_fwd, current_fwd = heapq.heappop(open_set_fwd)
                search_counter += 1
                if search_counter > max_search_area:
                    logger.warning("Pathfinding aborted: search area too large.")
                    return None
                if current_fwd in closed_set_fwd:
                    continue
                closed_set_fwd.add(current_fwd)

                if current_fwd in closed_set_bwd:
                    meeting_node = current_fwd
                    break

                neighbors = self.get_neighbors(current_fwd)
                for nx, ny in neighbors:
                    neighbor = (nx, ny)
                    if neighbor in closed_set_fwd:
                        continue
                    cell_value = self.map[start_map_id].get(neighbor, 0)
                    if cell_value in {-1}:
                        continue
                    if self.is_transition_position(start_map_id, nx, ny):
                        continue
                    curiosity_penalty = 0 - self.curiosity_map[start_map_id].get(neighbor, 1.0)
                    tentative_g = g_score_fwd[current_fwd] + 1 + curiosity_penalty
                    if tentative_g < g_score_fwd[neighbor]:
                        came_from_fwd[neighbor] = current_fwd
                        g_score_fwd[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, (goal_x, goal_y))
                        heapq.heappush(open_set_fwd, (f_score, neighbor))

            if open_set_bwd:
                current_f_bwd, current_bwd = heapq.heappop(open_set_bwd)
                search_counter += 1
                if search_counter > max_search_area:
                    logger.warning("Pathfinding aborted: search area too large.")
                    return None
                if current_bwd in closed_set_bwd:
                    continue
                closed_set_bwd.add(current_bwd)

                if current_bwd in closed_set_fwd:
                    meeting_node = current_bwd
                    break

                neighbors = self.get_neighbors(current_bwd)
                for nx, ny in neighbors:
                    neighbor = (nx, ny)
                    if neighbor in closed_set_bwd:
                        continue
                    cell_value = self.map[start_map_id].get(neighbor, 0)
                    if cell_value == -1:
                        continue
                    if self.is_transition_position(start_map_id, nx, ny):
                        continue   
                    curiosity_penalty = 1.0 - self.curiosity_map[start_map_id].get(neighbor, 1.0)
                    tentative_g = g_score_bwd[current_bwd] + 1 + curiosity_penalty
                    if tentative_g < g_score_bwd[neighbor]:
                        came_from_bwd[neighbor] = current_bwd
                        g_score_bwd[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, (start_x, start_y))
                        heapq.heappush(open_set_bwd, (f_score, neighbor))

        if not meeting_node:
            return None

        path_fwd = []
        current = meeting_node
        while current != (start_x, start_y):
            if self.map[start_map_id].get(current, 0) not in [0, -1] or current == meeting_node:
                path_fwd.append(current)
            current = came_from_fwd.get(current)
            if current is None:
                return None
        path_fwd.append((start_x, start_y))
        path_fwd.reverse()

        path_bwd = []
        current = meeting_node
        while current != (goal_x, goal_y):
            current = came_from_bwd.get(current)
            if current is None:
                return None
            if self.map[start_map_id].get(current, 0) not in [0, -1] or current == (goal_x, goal_y):
                path_bwd.append(current)

        full_path = path_fwd + path_bwd
        full_path_mapped = [(start_map_id, pos[0], pos[1]) for pos in full_path]
        for pos in full_path_mapped:
            map_id, x, y = pos
            if self.map[map_id].get((x, y), 0) in [-1]:
                logger.critical(f"Path includes an obstacle or unexplored square at position ({x}, {y}) on map {map_id}.")
                return None
        logger.debug(f"Planned bidirectional path: {full_path_mapped}")
        return full_path_mapped if full_path_mapped else None

    def get_action_from_positions(self, current: Tuple[int, int, int], next_pos: Tuple[int, int, int]) -> int:
        _, cx, cy = current
        _, nx, ny = next_pos
        dx = nx - cx
        dy = ny - cy
        logger.debug(f"Current position: ({cx}, {cy}), Next position: ({nx}, {ny}), dx: {dx}, dy: {dy}")
        if dx in [-1, 1] and dy in [-1, 1]:
            action = random.choice([1, 2, 3, 4])
            logger.critical(f"Invalid diagonal move from {current} to {next_pos}. Fallback action: {action}")

        if dx == 1 and dy == 0:
            action = 1  # Move Right
        elif dx == -1 and dy == 0:
            action = 2  # Move Left
        elif dx == 0 and dy == -1:
            action = 3  # Move Up
        elif dx == 0 and dy == 1:
            action = 4  # Move Down
        else:

            logger.critical(f"Unexpected movement from {current} to {next_pos}. Resetting frontiers.")
            self.frontiers = set()
            if self.current_frontier:
                self.old_frontier = self.current_frontier
            self.current_frontier = None
            self.current_path = []
            action = self.ppo_action

        logger.debug(f"Selected action: {action}")
        return action

    def check_collision(self, position: Tuple[int, int, int]) -> bool:
        attempted_map_id, attempted_x, attempted_y = position

        if attempted_map_id not in self.map_offsets:
            logger.error(f"Map ID {attempted_map_id} not found in map_offsets.")
            return True

        offset_x, offset_y = self.map_offsets[attempted_map_id]
        attempted_grid_x = attempted_x + offset_x
        attempted_grid_y = attempted_y + offset_y

        ash_map_id, ash_grid_x, ash_grid_y = self.get_ash_position_grid()

        if attempted_map_id != ash_map_id:
            logger.debug(f"Attempted to move to a different map ID: {attempted_map_id} (Current Map ID: {ash_map_id})")
            return True

        cell_value = self.map[attempted_map_id].get((attempted_grid_x, attempted_grid_y), 0)
        if cell_value == -1:
            logger.debug(f"Attempted grid position ({attempted_grid_x}, {attempted_grid_y}) is an obstacle.")
            return True

        if (attempted_grid_x, attempted_grid_y) != (ash_grid_x, ash_grid_y):
            logger.debug(
                f"Collision detected: Attempted Grid Position ({attempted_grid_x}, {attempted_grid_y}) differs from Ash's Grid Position ({ash_grid_x}, {ash_grid_y}).")
            return True
        else:
            logger.debug("No collision detected.")
            return False

    def get_attempted_position(self, position: Tuple[int, int, int], action: int) -> Tuple[int, int, int]:
        map_id, x, y = position
        dx, dy = {1: (1, 0), 2: (-1, 0), 3: (0, -1), 4: (0, 1)}.get(action, (0, 0))
        attempted_x = x + dx
        attempted_y = y + dy
        return (map_id, attempted_x, attempted_y)

    def get_ash_position_grid(self) -> Tuple[int, int, int]:
        map_id, x, y = self.ash_position
        self.initialize_map_offset(map_id, x, y)
        offset_x, offset_y = self.map_offsets[map_id]
        grid_x = x + offset_x
        grid_y = y + offset_y
        logger.debug(f"Ash's grid position: Map ID {map_id}, Grid Position ({grid_x}, {grid_y})")
        return (map_id, grid_x, grid_y)

    def increment_collision_count(self, position):
        map_id, x, y = position
        self.collision_counts[(map_id, x, y)] += 1

    def get_collision_count(self, position):
        map_id, x, y = position
        return self.collision_counts[(map_id, x, y)]

    def reset_collision_count(self, position):
        map_id, x, y = position
        del self.collision_counts[(map_id, x, y)]
    