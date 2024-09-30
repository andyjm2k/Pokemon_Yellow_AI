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

# --------------------- Logging Configuration ---------------------
# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log message format
    handlers=[
        logging.StreamHandler()  # Output logs to the console
        # Add logging.FileHandler('app.log') here if you want to log to a file
    ]
)

# Create a logger for this module
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------


class GbaGame(Env):
    def __init__(self, max_episodes=500000):
        super().__init__()
        self.frame_stack = deque(maxlen=1)
        self.frame_skip = 1  # Number of frames to skip
        # Adjust observation space to 3D for CNN compatibility
        self.observation_space = Box(low=0, high=255, shape=(120, 120, 3), dtype=np.uint8)
        self.action_space = Discrete(6)
        self.pyboy = PyBoy('ROMs/Pokemon_Yellow.gbc', window_type="headless",
                           scale=3)

        # **Initialize Map Visualization Parameters Before Reset**
        self.map_size = 100  # Define the size of the map grid
        self.map_offset = self.map_size // 2  # Offset to handle negative coordinates
        self.cell_size = 5  # Size of each cell in pixels for visualization
        self.map_image = np.ones((self.map_size * self.cell_size, self.map_size * self.cell_size, 3), dtype=np.uint8) * 255  # White background

        # Initialize SLAM data structures
        self.map = defaultdict(lambda: defaultdict(int))  # 2D grid map: 0=unknown, 1=free, -1=obstacle
        self.encounter_positions = defaultdict(set)
        self.transitions = defaultdict(set)  # **Added: To track map transitions**
        self.current_map_id = None
        self.ash_position = None
        self.ash_old_position = None
        self.frontiers = set()  # Set of frontier cells as tuples (map_id, x, y)
        self.current_path = []  # Current planned path as list of positions
        self.target_frontier = None  # Current target frontier
        self.current_frontier = None

        # **Now, Call reset_game_state After All Necessary Attributes Are Initialized**
        self.reset_game_state()

        # Initialize other attributes
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
        self.pokemon_found_list = [-1]
        self.pokemon_caught_list = [-1]
        self.level_progress_pct = 0.0
        self.ash_is_moving = -1
        self.ash_stuck_counter = 0
        self.is_battling_fl = False
        self.new_total_hp = 0
        self.first_loc_check = 0
        self.new_enemy_hp = 0
        self.agent_id = random.randint(1, 1000000)
        self.first_episode = True
        self.previous_enemy_lvl = 1000
        self.global_goal = ''
        self.pkm_fnd = -1
        self.pkm_cau = -1
        self.pokemon_tally = 0
        self.last_action = None  # Keep track of the last action taken
        self.previous_action = None  # Initialize previous action
        self.preferred_action = None  # Keep track of the preferred action direction
        self.ash_loc_history = deque(maxlen=50)  # For detecting if Ash is stuck
        self.visited_frontiers = set()
        print('STARTED AGENT: ', self.agent_id)

    def reset_game_state(self):
        self.total_reward = 0
        self.episode_length = 0
        self.current_score = 0
        self.level_progress = [-1]
        self.pokemon_found_list = [-1]
        self.pokemon_caught_list = [-1]
        self.level_progress_pct = 0.0
        self.ash_is_moving = -1
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
        self.last_action = None  # Reset last action
        self.previous_action = None  # Reset previous action
        self.preferred_action = None  # Reset preferred action
        self.ash_loc_history = deque(maxlen=50)  # Reset Ash's position history

        # Reset SLAM data structures
        self.map = defaultdict(lambda: defaultdict(int))
        self.encounter_positions = defaultdict(set)
        self.transitions = defaultdict(set)  # **Added: Reset transitions**
        self.current_map_id = None
        self.ash_position = None
        self.ash_old_position = None
        self.frontiers = set()
        self.current_path = []
        self.target_frontier = None
        self.current_frontier = None
        self.visited_frontiers = set()
        # Reset Map Image to White Background
        self.map_image = np.ones((self.map_size * self.cell_size, self.map_size * self.cell_size, 3), dtype=np.uint8) * 255  # White background

    def step(self, action):
        total_reward = 0
        done = False
        info = {}
        for _ in range(self.frame_skip):
            if not done:
                # Determine the goal
                self.get_goal()
                self.battling()  # Update self.is_battling_fl

                # Determine the action
                if self.global_goal == 'blue' and not self.is_battling_fl:
                    action = self.slam_action()
                    # Set last_action regardless of the goal
                    self.last_action = action

                    # Execute the action
                    self.execute_action(self.last_action)

                    # Release the action key
                    self.release_all_keys()

                    # Tick the emulator to allow game state to update
                    self.pyboy.tick(1, False)

                    # Update Ash's position
                    self.ash_old_position = self.ash_position
                    self.ash_position = self.get_ash_position()
                    self.ash_loc_history.append(self.ash_position)

                    # Update SLAM map
                    self.update_slam_map()

                else:
                    self.last_action = action
                    # Execute the action
                    self.execute_action(self.last_action)
                    # Release the action key
                    self.release_all_keys()
                    # Tick the emulator to allow game state to update
                    self.pyboy.tick(1, False)
                    if not self.is_battling_fl:
                        # Update Ash's position
                        self.ash_position = self.get_ash_position()
                        self.ash_loc_history.append(self.ash_position)

                        # Update SLAM map
                        self.update_slam_map()

                # Update the frame stack
                self.update_frame_stack()

                # Render the game and the SLAM map
                self.render()

                # Update counters and check for termination
                self.current_step += 1
                self.episode_length += 1
                reward, done = self.calculate_reward_and_done(self.last_action)
                truncated = self.truncated
                if truncated:
                    info = {}
        return self.get_stacked_observation(), reward, done, truncated, info

    def reset(self, seed=None, options=None):
        save_filename = f"ROMs/save_state_agent_{self.agent_id}.state"
        with open(save_filename, "wb") as save_file_object:
            self.pyboy.save_state(save_file_object)
        if self.pyboy_counter == 10000:
            self.pyboy.stop()
            del self.pyboy
            self.pyboy_counter = 0
            self.pyboy = PyBoy('ROMs/Pokemon_Yellow.gbc', window_type="headless",
                               scale=3, game_wrapper=False)
        if seed:
            np.random.seed(seed)
        self.reset_game_state()
        self.reset_game_in_gui()
        if self.initial_observation:
            self.get_goal()
            l_obs = self.get_observation()
            self.frame_stack.extend([l_obs] * self.frame_stack.maxlen)
        self.pyboy_counter += 1
        # Initialize SLAM variables
        self.current_map_id = self.get_map_id()
        self.ash_position = self.get_ash_position()
        self.add_to_map(self.ash_position, 1)  # Mark starting position as free space
        self.detect_frontiers()
        # Update Map Image with the starting position
        self.update_map_image()
        return self.get_stacked_observation(), {}

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

        # Display the game screen
        label = str(self.agent_id)
        cv2.imshow(label, last)  # Display the image

        # Render the SLAM map
        self.render_map()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def render_map(self):
        """
        Renders the SLAM map with:
            - Explored positions as black blocks
            - Obstacles as green blocks
            - Frontier positions as blue blocks
            - Pokémon encounters as cyan blocks
            - Transitions as medium brown blocks
            - Ash's current position as a red block
        """
        if not self.ash_position:
            logger.warning("Ash's position is not set.")
            return

        map_id, ash_x_game, ash_y_game = self.ash_position

        # Convert game coordinates to map grid coordinates
        ash_x = ash_x_game + self.map_offset
        ash_y = ash_y_game + self.map_offset

        # Validate grid coordinates
        if not (0 <= ash_x < self.map_size and 0 <= ash_y < self.map_size):
            logger.warning(f"Ash's grid position ({ash_x}, {ash_y}) is out of map bounds.")
            return

        # Reset the map image to white
        self.map_image[:] = 255

        # Draw explored positions as black blocks and obstacles as green
        for (x, y), value in self.map[map_id].items():
            top_left = (x * self.cell_size, y * self.cell_size)
            bottom_right = ((x + 1) * self.cell_size, (y + 1) * self.cell_size)
            color = (0, 0, 0) if value == 1 else (0, 255, 0) if value == -1 else None
            if color:
                cv2.rectangle(self.map_image, top_left, bottom_right, color, -1)

        # Draw frontiers as blue blocks
        for frontier in list(self.frontiers):
            f_map_id, fx, fy = frontier
            if f_map_id != map_id:
                continue  # Only render frontiers in the current map
            # Convert frontier game coordinates to grid coordinates
            fx_grid = fx + self.map_offset
            fy_grid = fy + self.map_offset
            top_left = (fx_grid * self.cell_size, fy_grid * self.cell_size)
            bottom_right = ((fx_grid + 1) * self.cell_size, (fy_grid + 1) * self.cell_size)
            cv2.rectangle(self.map_image, top_left, bottom_right, (255, 0, 0), -1)  # Blue

        # Draw Pokémon encounters as cyan blocks for the current map
        for (x, y) in self.encounter_positions[map_id]:
            top_left = (x * self.cell_size, y * self.cell_size)
            bottom_right = ((x + 1) * self.cell_size, (y + 1) * self.cell_size)
            cv2.rectangle(self.map_image, top_left, bottom_right, (255, 255, 0), -1)  # Cyan

        # **Added: Draw transitions as medium brown blocks**
        brown_color = (42, 42, 165)  # BGR for medium brown (RGB: (165, 42, 42))
        for (x, y) in self.transitions[map_id]:
            top_left = (x * self.cell_size, y * self.cell_size)
            bottom_right = ((x + 1) * self.cell_size, (y + 1) * self.cell_size)
            cv2.rectangle(self.map_image, top_left, bottom_right, brown_color, -1)  # Medium Brown

        # Draw Ash's current position as a red block on top of other elements
        top_left = (ash_x * self.cell_size, ash_y * self.cell_size)
        bottom_right = ((ash_x + 1) * self.cell_size, (ash_y + 1) * self.cell_size)
        cv2.rectangle(self.map_image, top_left, bottom_right, (0, 0, 255), -1)  # Red

        # Optional: Add a border around Ash's position for better visibility
        cv2.rectangle(self.map_image, top_left, bottom_right, (255, 0, 0), 1)  # Blue border

        # Display the map
        cv2.imshow("SLAM Map", self.map_image)

        # Handle window closure gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def close(self):
        cv2.destroyAllWindows()
        self.pyboy.stop()

    def get_observation(self):
        self.pyboy.memory[0xd31e] = 99  # Example modification if needed
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
        # Release all keys before pressing a new one
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
            # Determine the number of ticks needed
            ticks_needed = 1  # Default to 1 tick
            if action in [1, 2, 3, 4] and not self.is_battling_fl:
                if self.previous_action is not None:
                    opposite_actions = {1: 2, 2: 1, 3: 4, 4: 3}
                    if action == opposite_actions.get(self.previous_action):
                        # Opposite direction
                        # print("Opposite direction triggered")
                        ticks_needed = 2
            # Press the key
            self.pyboy.send_input(key_event)
            # Hold the key down for the required number of ticks
            for _ in range(ticks_needed):
                self.pyboy.tick(1, True)
        # Update previous_action
        if action in [1, 2, 3, 4]:
            self.previous_action = action

    def update_frame_stack(self):
        observation = self.get_observation()
        self.frame_stack.append(observation)

    def calculate_reward_and_done(self, action):
        goal = self.global_goal
        reward = self.calculate_reward_basic(action, goal)
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

    def calculate_reward_basic(self, action, goal):
        reward = 0
        if self.timed_out():
            self.truncated = True
            print("Episode timed out.")
        # Reward for exploring new positions
        loc = self.ash_position[1:]  # Exclude map_id
        if len(loc) < 2:
            print(f"Error: ash_position has insufficient elements: {self.ash_position}")
            return reward  # Early exit to prevent further errors

        map_x = loc[0] + self.map_offset  # Corrected index
        map_y = loc[1] + self.map_offset  # Corrected index

        if self.map[self.ash_position[0]].get((map_x, map_y), 0) == 1:
            # Already visited; no additional reward
            pass
        else:
            if goal == 'blue':
                reward += 0.001
                logger.info(f"Agent ID: {self.agent_id} | Goal: {goal} | Ash got explore reward")
            self.map[self.ash_position[0]][(map_x, map_y)] = 1  # Mark as visited

        if self.is_ash_stuck():
            if self.ash_stuck_counter >= 3000:
                reward -= 0.01
                if self.ash_stuck_counter >= 5000:
                    reward -= 10
                    logger.info(f"Agent ID: {self.agent_id} | Goal: {goal} | Ash not unstuck, exiting")
        else:
            self.ash_stuck_counter = 0

        # Other reward calculations
        if self.chk_battling():
            pkm_f = self.pkm_fnd
            if pkm_f not in self.pokemon_found_list:
                self.pokemon_found_list.append(pkm_f)
                if goal == 'magenta':
                    reward += 1
                    logger.info(f"Agent ID: {self.agent_id} | Goal: {goal} | Ash discovered a Pokemon")
            pkm_c = self.pkm_cau
            if pkm_c not in self.pokemon_caught_list:
                v_1 = self.pokemon_caught()
                if v_1 > self.pokemon_tally:
                    self.pokemon_caught_list.append(pkm_c)
                    self.pokemon_tally = v_1
                    if goal == 'magenta':
                        reward += 10
                        logger.info(f"Agent ID: {self.agent_id} | Goal: {goal} | Ash caught a Pokemon")
            current_score = self.get_score()
            if current_score == 10:
                if goal == 'red':
                    reward += current_score
                    logger.info(f"Agent ID: {self.agent_id} | Goal: {goal} | Reward: Total Levels went up by {current_score / 10}")
        if self.battling():
            if not self.is_battling_fl:
                if goal == 'red':
                    reward += 0.1
                    logger.info(f"Agent ID: {self.agent_id} | Goal: {goal} | Battling")
        if self.did_damage():
            if self.new_enemy_hp == 0:
                if goal == 'red':
                    reward += 1
                    logger.info(f"Agent ID: {self.agent_id} | Goal: {goal} | Beat Pokemon")
                else:
                    reward += 0
                    logger.info(f"Agent ID: {self.agent_id} | Goal: {goal} | Did damage")
        if self.did_hp_drop():
            if goal == 'red':
                reward += 0
                logger.info(f"Agent ID: {self.agent_id} | Goal: {goal} | HP dropped")
        if self.level_did_progress() > 0:
            if goal == 'blue':
                logger.info(f"Agent ID: {self.agent_id} | Goal: {goal} | Level progressed = {self.level_progress}")
                reward += 10
        return reward

    def detect_ash_faint(self):
        # Placeholder for faint detection logic
        return False

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
        for _ in range(self.wait_frames):
            self.pyboy.tick()
        if self.first_episode:
            selected_filename = "ROMs/Pokemon_Yellow.gbc.state.old2"
            self.first_episode = False
        else:
            selected_filename = "ROMs/Pokemon_Yellow.gbc.state.old2"
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

    def found_flag(self):
        return False

    def chk_battling(self):
        return self.pyboy.memory[0xd056] != 0

    def battling(self):
        battle_state = self.pyboy.memory[0xd056]
        if battle_state != 0:
            if not self.is_battling_fl:
                self.is_battling_fl = True
                self.new_enemy_hp = self.pyboy.memory[0xcfe6]
                self.previous_enemy_lvl = self.pyboy.memory[0xcff2]
                # print("Battle started.")

                # Record the encounter position under the current map ID
                current_grid_pos = self.get_ash_position_grid()
                self.encounter_positions[current_grid_pos[0]].add((current_grid_pos[1], current_grid_pos[2]))
                logger.info(
                    f"Battle started at Map ID {current_grid_pos[0]}, Position ({current_grid_pos[1]}, {current_grid_pos[2]}).")

                return True
        else:
            if self.is_battling_fl:
                # print("Battle ended.")
                self.is_battling_fl = False
                return False

    def timed_out(self):
        return self.episode_length > self.max_episodes

    def is_ash_stuck(self):
        # Do not check if Ash is stuck when battling
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
            if self.first_loc_check == 0:
                self.level_progress.append(v_1)
                self.first_loc_check = 1
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
            else:
                return 0
        else:
            return 0

    def new_pokemon_caught(self):
        if self.chk_battling():
            v_1 = self.pyboy.memory[0xcfd9]
            if v_1 not in self.pokemon_caught_list:
                self.pkm_cau = v_1
                return 1
            else:
                return 0
        else:
            return 0

    def get_goal(self):
        total_lvls = self.current_score
        enemy_lvl = self.previous_enemy_lvl if self.previous_enemy_lvl > 0 else 1
        new_pkm_fnd = self.new_pokemon_found()
        new_pkm_cau = self.new_pokemon_caught()
        num_poke_balls = self.pyboy.memory[0xd31e]

        too_strong = total_lvls > (enemy_lvl * 10)
        too_many_steps = self.ash_stuck_counter > 3000
        no_poke_balls = num_poke_balls == 0

        # Check if Ash is battling
        battle_state = self.pyboy.memory[0xd056]
        player_control = self.pyboy.memory[0xc507]
        if battle_state == 2:
            goal = 'red'  # Battle the trainer
        elif battle_state == 1:
            if new_pkm_fnd == 1 or new_pkm_cau == 1:
                goal = 'magenta'  # Catch new Pokemon
            elif too_strong:
                goal = 'green'  # Explore new areas
            else:
                goal = 'red'  # Battle wild Pokemon to gain experience
        else:
            # Not in battle
            if player_control == 126 and battle_state == 0:
                goal = 'red'
            elif too_many_steps:
                goal = 'blue'  # Explore location
            elif not too_strong:
                goal = 'blue'  # Power up Pokemon by battling
            elif too_strong and battle_state == 0:
                goal = 'blue'
            else:
                goal = 'green'  # Default to exploring

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
        # Get Ash's current position
        y_coord = self.pyboy.memory[0xd360]
        x_coord = self.pyboy.memory[0xd361]
        map_id = self.pyboy.memory[0xd35d]
        return (map_id, x_coord, y_coord)

    def add_to_map(self, position, value):
        map_id, x, y = position
        grid_x = x + self.map_offset
        grid_y = y + self.map_offset
        current_value = self.map[map_id].get((grid_x, grid_y), 0)

        if value == -1:
            if current_value != 1:
                self.map[map_id][(grid_x, grid_y)] = -1  # Mark as obstacle
                logger.info(f"Mapping updated: Map ID {map_id}, Position ({grid_x}, {grid_y}) set to Obstacle (-1)")
            else:
                logger.info(f"Attempted to mark as obstacle, but position ({grid_x}, {grid_y}) is already free.")
        elif value == 1:
            if current_value != 1:
                self.map[map_id][(grid_x, grid_y)] = 1  # Mark as free
                logger.info(f"Mapping updated: Map ID {map_id}, Position ({grid_x}, {grid_y}) set to Free (1)")
            else:
                logger.info(f"Position ({grid_x}, {grid_y}) is already marked as free.")
        # No action needed for value == 0 (unknown)

    def update_map_image(self, x: Optional[int] = None, y: Optional[int] = None, value: Optional[int] = None):
        """
        Updates the map_image based on the new information.
        If x and y are provided, update only that cell.
        Otherwise, redraw the entire map.
        """
        map_id, ash_x, ash_y = self.ash_position
        if x is not None and y is not None and value is not None:
            top_left = (x * self.cell_size, y * self.cell_size)
            bottom_right = ((x + 1) * self.cell_size, (y + 1) * self.cell_size)
            if value == 1:
                cv2.rectangle(self.map_image, top_left, bottom_right, (0, 0, 0), -1)  # Black for free
            elif value == -1:
                cv2.rectangle(self.map_image, top_left, bottom_right, (0, 255, 0), -1)  # Green for obstacles
        else:
            # Redraw entire map
            self.map_image[:] = 255  # Reset to white
            for (cell_x, cell_y), cell_value in self.map[map_id].items():
                top_left = (cell_x * self.cell_size, cell_y * self.cell_size)
                bottom_right = ((cell_x + 1) * self.cell_size, (cell_y + 1) * self.cell_size)
                if cell_value == 1:
                    cv2.rectangle(self.map_image, top_left, bottom_right, (0, 0, 0), -1)  # Black
                elif cell_value == -1:
                    cv2.rectangle(self.map_image, top_left, bottom_right, (0, 255, 0), -1)  # Green for obstacles

    def update_slam_map(self):
        """
        Update the SLAM map based on Ash's movement.
        """
        if len(self.ash_loc_history) < 50:
            # Not enough history to determine movement
            return

        previous_position = self.ash_loc_history[-24]
        current_position = self.ash_position

        # **Added: Detect map transitions**
        if self.ash_old_position and self.ash_position:
            previous_map_id = self.ash_old_position[0]
            current_map_id = self.ash_position[0]
            if previous_map_id != current_map_id:
                # Transition detected
                prev_map_id, prev_x, prev_y = self.ash_old_position
                new_map_id, new_x, new_y = self.ash_position

                # Convert to grid coordinates
                prev_grid_x = prev_x + self.map_offset
                prev_grid_y = prev_y + self.map_offset
                new_grid_x = new_x + self.map_offset
                new_grid_y = new_y + self.map_offset

                # **Mark the last position on the old map as a transition**
                self.transitions[prev_map_id].add((prev_grid_x, prev_grid_y))
                logger.info(f"Transition detected: Exiting Map ID {prev_map_id} at ({prev_grid_x}, {prev_grid_y})")

                # **Mark the first position on the new map as a transition**
                self.transitions[new_map_id].add((new_grid_x, new_grid_y))
                logger.info(f"Transition detected: Entering Map ID {new_map_id} at ({new_grid_x}, {new_grid_y})")

        if previous_position == current_position:
            # Ash did not move; likely a collision
            if self.last_action in [1, 2, 3, 4]:
                attempted_position = self.get_attempted_position(previous_position, self.last_action)
                collision = self.check_collision(attempted_position)
                if collision:
                    self.add_to_map(attempted_position, -1)  # Mark as obstacle
                    self.current_path = []  # Clear the current path due to collision
                    logger.info(f"Collision detected at {attempted_position}. Clearing current path.")
                else:
                    logger.warning(f"Unable to move, but no collision detected at {attempted_position}.")
        else:
            # Ash moved successfully
            self.add_to_map(current_position, 1)  # Mark as free space
            logger.info(f"Ash moved to x={current_position[1]+self.map_offset} y={current_position[2]+self.map_offset}")

        # Detect and update frontiers
        self.detect_frontiers()

        # Check if Ash has reached the current frontier
        self.check_target_frontier_reached()

    def detect_frontiers(self):
        """
        Detect and update the set of frontier cells.
        A frontier cell is a free cell that has at least one unknown neighbor.
        Only the nearest frontier is retained in self.frontiers.
        """
        map_id, x, y = self.ash_position
        ash_grid_x = x + self.map_offset
        ash_grid_y = y + self.map_offset

        possible_frontiers = []

        # Iterate through all free cells to find potential frontiers
        for (cell_x, cell_y), value in self.map[map_id].items():
            if (
                    value == 1 and  # Free cell
                    (map_id, cell_x, cell_y) != self.current_frontier and  # Not the current target frontier
                    (cell_x, cell_y) != (ash_grid_x, ash_grid_y) and  # Not Ash's current position
                    (map_id, cell_x, cell_y) not in self.visited_frontiers  # Not already visited
            ):
                neighbors = self.get_neighbors((cell_x, cell_y))
                for nx, ny in neighbors:
                    if self.map[map_id].get((nx, ny), 0) == 0:  # Unknown neighbor
                        possible_frontiers.append((map_id, cell_x, cell_y))
                        break  # No need to check other neighbors for this cell

        if possible_frontiers:
            # Determine Ash's current grid position
            ash_pos = (ash_grid_x, ash_grid_y)

            # Function to calculate Manhattan distance
            def manhattan_distance(frontier):
                _, fx, fy = frontier
                return abs(fx - ash_grid_x) + abs(fy - ash_grid_y)

            # Find the frontier with the smallest distance to Ash
            nearest_frontier = min(possible_frontiers, key=manhattan_distance)

            # Update self.frontiers to contain only the nearest frontier
            self.frontiers = {nearest_frontier}
            logger.info(
                f"New frontier set to Map ID {nearest_frontier[0]}, Position ({nearest_frontier[1]}, {nearest_frontier[2]})")
        else:
            # No frontiers found; clear the frontiers set
            self.frontiers = set()
            logger.info("No frontiers detected.")

    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get 4-connected neighbors of a given position with boundary checks.
        """
        x, y = position
        neighbors = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:  # Only cardinal directions
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                neighbors.append((nx, ny))
            else:
                logger.debug(f"Neighbor out of bounds: ({nx}, {ny})")
        return neighbors

    def check_target_frontier_reached(self):
        """
        Check if Ash has reached the current frontier.
        If reached, add it to visited_frontiers and reset the current frontier.
        """
        if self.current_frontier:
            if self.get_ash_position_grid() == self.current_frontier:
                logger.info(f"Ash has reached the current frontier at {self.current_frontier}")
                # Add the reached frontier to visited_frontiers
                self.visited_frontiers.add(self.current_frontier)
                # Reset the current frontier and current path
                self.current_frontier = None
                self.current_path = []  # Clear the current path as it's completed

    def slam_action(self) -> int:
        """
        Determine the next action based on Frontier-Based SLAM.
        Follows the existing path if available; otherwise, plans a new path to the current frontier.
        """
        if self.current_path:
            # Follow the existing path
            next_position = self.current_path.pop(0)
            logger.info(f"Following existing path: Next position {next_position}")
            return self.get_action_from_positions(self.get_ash_position_grid(), next_position)
        else:
            if not self.current_frontier:
                # No frontiers left; exploration complete
                logger.info("No frontiers left; exploration complete. Taking random action.")
                return random.choice([1, 2, 3, 4])  # Random move to continue

            # Plan path to the current frontier
            path = self.plan_path(self.ash_position, self.current_frontier)
            if path is None or len(path) < 2:
                # Cannot find a path to the frontier
                logger.warning("Cannot find a path to the frontier. Taking random action.")
                self.visited_frontiers.add(self.current_frontier)  # Mark as visited to prevent re-selection
                self.current_frontier = None  # Reset the current frontier
                return random.choice([1, 2, 3, 4])  # Random move to continue

            # Store the current path, excluding the starting position
            self.current_path = path[1:]
            logger.info(f"Path planned: {self.current_path}")
            # Determine the next action from the path
            next_position = self.current_path.pop(0)
            logger.info(f"self.ash_position = {self.get_ash_position_grid()}, next_position = {next_position}")
            return self.get_action_from_positions(self.get_ash_position_grid(), next_position)

    def select_nearest_frontier(self) -> Optional[Tuple[int, int, int]]:
        """
        Select the nearest frontier using BFS.
        """
        ash_map_id, ash_x_grid, ash_y_grid = self.get_ash_position_grid()
        visited = set()
        queue = deque()
        queue.append((ash_map_id, ash_x_grid, ash_y_grid, 0))  # (map_id, x, y, distance)
        visited.add((ash_map_id, ash_x_grid, ash_y_grid))

        while queue:
            current = queue.popleft()
            current_map_id, cx, cy, dist = current

            if (current_map_id, cx, cy) in self.frontiers:
                logger.info(f"Nearest frontier found at Map ID {current_map_id}, Position ({cx}, {cy}) with distance {dist}")
                return (current_map_id, cx, cy)

            neighbors = self.get_neighbors((cx, cy))
            for nx, ny in neighbors:
                if (nx, ny) in visited:
                    continue
                cell_value = self.map[current_map_id].get((nx, ny), 0)
                if cell_value == 1:
                    queue.append((current_map_id, nx, ny, dist + 1))
                    visited.add((nx, ny))
                else:
                    logger.debug(f"Skipping Map ID {current_map_id}, Position ({nx}, {ny}) - Nearest Cell Value: {cell_value}")

        return None  # No frontier found

    def plan_path(self, start: Tuple[int, int, int], goal: Tuple[int, int, int]) -> Optional[
        List[Tuple[int, int, int]]]:
        """
        Plan a path from start to goal using A*.
        Returns a list of positions from start to goal.
        """
        start_map_id, start_x, start_y = start
        start_x += self.map_offset
        start_y += self.map_offset
        goal_map_id, goal_x, goal_y = goal
        logger.info(f"start {start}, goal {goal}")

        if start_map_id != goal_map_id:
            logger.warning("Path planning across multiple maps is not supported.")
            return None

        open_set = []
        heapq.heappush(open_set, (0, (start_x, start_y)))
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[(start_x, start_y)] = 0

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

        while open_set:
            current_f, current = heapq.heappop(open_set)
            if current == (goal_x, goal_y):
                break
            neighbors = self.get_neighbors(current)
            for nx, ny in neighbors:
                if (nx, ny) in came_from:
                    continue
                cell_value = self.map[start_map_id].get((nx, ny), 0)
                if cell_value == 1:
                    tentative_g = g_score[current] + 1
                    if tentative_g < g_score[(nx, ny)]:
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative_g
                        f_score = tentative_g + heuristic((nx, ny), (goal_x, goal_y))
                        heapq.heappush(open_set, (f_score, (nx, ny)))
                else:
                    logger.debug(f"Skipping Map ID {start_map_id}, Position ({nx}, {ny}) - Cell Value: {cell_value}")

        else:
            # Goal not reached
            logger.warning("A* Path planning failed: Goal not reached.")
            return None

        # Reconstruct path
        path = []
        current = (goal_x, goal_y)
        while current != (start_x, start_y):
            path.append(current)
            current = came_from[current]
        path.append((start_x, start_y))
        path.reverse()

        # Convert to map_id-based positions
        full_path = [(start_map_id, pos[0], pos[1]) for pos in path]
        logger.info(f"Path planned: {full_path}")

        # Validate path steps
        for i in range(1, len(full_path)):
            current = full_path[i - 1]
            next_step = full_path[i]
            _, cx, cy = current
            _, nx, ny = next_step
            dx = nx - cx
            dy = ny - cy
            if (abs(dx) + abs(dy)) != 1:
                logger.error(f"Invalid step in path: {current} to {next_step}")
                return None  # Invalid path

        return full_path

    def get_action_from_positions(self, current: Tuple[int, int, int], next_pos: Tuple[int, int, int]) -> int:
        """
        Determine the action to take to move from current to next_pos.
        Actions:
            1: Move Right
            2: Move Left
            3: Move Up
            4: Move Down
            0: Press A
            5: Press B
        """
        logger.info(f"current={current}, next_pos={next_pos}")
        _, cx, cy = current
        _, nx, ny = next_pos
        dx = nx - cx
        dy = ny - cy

        if dx == 1 and dy == 0:
            return 1  # Move Right
        elif dx == -1 and dy == 0:
            return 2  # Move Left
        elif dx == 0 and dy == -1:
            return 3  # Move Up
        elif dx == 0 and dy == 1:
            return 4  # Move Down
        else:
            logger.error(f"Invalid movement from {current} to {next_pos}. Pressing A or B.")
            return random.choice([0, 5])  # Press A or B randomly

    def check_collision(self, position: Tuple[int, int, int]) -> bool:
        """
        Verify if the attempted position is blocked by comparing the attempted grid position
        with Ash's current grid position.
        """
        attempted_map_id, attempted_x, attempted_y = position
        ash_map_id, ash_x, ash_y = self.get_ash_position_grid()

        # If attempting to move to a different map, assume collision
        if attempted_map_id != ash_map_id:
            return True

        # Check if the attempted position matches Ash's current position
        if (attempted_x, attempted_y) != (ash_x, ash_y):
            return True  # Collision detected
        else:
            return False  # No collision

    def get_attempted_position(self, position: Tuple[int, int, int], action: int) -> Tuple[int, int, int]:
        """
        Calculate the position Ash attempted to move to based on the action.
        """
        map_id, x, y = position  # x and y are grid coordinates (0 to map_size-1)
        dx, dy = {1: (1, 0), 2: (-1, 0), 3: (0, -1), 4: (0, 1)}.get(action, (0, 0))
        attempted_x = x + dx
        attempted_y = y + dy

        # Clamp to valid grid boundaries
        attempted_x = max(0, min(attempted_x, self.map_size - 1))
        attempted_y = max(0, min(attempted_y, self.map_size - 1))

        return (map_id, attempted_x, attempted_y)

    def get_ash_position_grid(self) -> Tuple[int, int, int]:
        """
        Returns Ash's current position in grid coordinates.
        """
        map_id, x, y = self.ash_position
        grid_x = x + self.map_offset
        grid_y = y + self.map_offset
        return (map_id, grid_x, grid_y)

    def game_to_grid(self, x: int, y: int) -> Tuple[int, int]:
        """
        Converts game coordinates to grid coordinates.
        """
        grid_x = x + self.map_offset
        grid_y = y + self.map_offset
        return (grid_x, grid_y)

    def grid_to_game(self, grid_x: int, grid_y: int) -> Tuple[int, int]:
        """
        Converts grid coordinates to game coordinates.
        """
        x = grid_x - self.map_offset
        y = grid_y - self.map_offset
        return (x, y)
