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
import asyncio
from collections import deque, defaultdict
from pyboy import PyBoy
import random
from typing import List, Tuple, Optional
from PIL import Image  # Ensure PIL is installed
import logging
import matplotlib.pyplot as plt
from llm_controller_gemini import LLMController

"""
Pokemon Yellow Reinforcement Learning Environment with Pathfinding

This module implements a custom Gym environment for training reinforcement learning agents to play
Pokemon Yellow. The environment integrates PyBoy (a Game Boy emulator) with Gymnasium's interface, 
allowing RL algorithms to interact with the game. The environment includes:

1. SLAM (Simultaneous Localization and Mapping) for navigation and exploration
2. Pathfinding algorithms for intelligent movement between locations
3. Battle detection and handling for Pokemon encounters
4. Reward systems based on different goals (exploration, battling, catching Pokemon)
5. Visualization tools for monitoring agent progress

Key Components:
- Map and frontier discovery for exploration
- A* pathfinding for efficient navigation
- Collision detection and handling
- Game state tracking (Pokemon party, items, location)
- Multiple goal modes (blue=exploration, red=battling, magenta=catching, green=random actions)

This environment is designed for research in reinforcement learning applied to complex game environments
with partial observability, long-term planning requirements, and diverse objectives.
"""

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
    """
    GbaGame Environment - A custom Gymnasium environment for Pokemon Yellow.
    
    This class provides the interface between reinforcement learning algorithms
    and the Pokemon Yellow game running on a PyBoy emulator. It implements the
    standard Gymnasium environment methods (step, reset, render, etc.) and adds
    game-specific functionality for navigation, battle handling, and Pokemon management.
    
    The environment handles:
    1. Game state management through PyBoy emulator
    2. Observation space (game screen images processed for the agent)
    3. Action space (controls that can be sent to the game)
    4. Reward calculation based on game events and progress
    5. SLAM-based mapping and navigation
    6. Path planning and automated exploration
    
    A key feature is the ability to switch between different goal modes:
    - 'blue': Exploration mode - rewards discovering new maps and areas
    - 'red': Battle mode - rewards engaging in and winning battles
    - 'magenta': Catch mode - rewards finding and catching new Pokemon
    - 'green': Random action mode - used when stuck or for exploration
    
    The environment builds a map of the game world as the agent plays,
    tracks frontiers (unexplored edges of the map), and uses pathfinding
    to navigate effectively through the game world.
    
    Dependencies:
    - PyBoy for Game Boy emulation
    - OpenCV for image processing
    - Gymnasium for RL environment interface
    - numpy for efficient array operations
    - A* pathfinding algorithm for navigation
    """
    def __init__(self, max_episodes=500000):
        """
        Initialize the Pokemon Yellow game environment.
        
        This method sets up the PyBoy emulator, configures the observation and action spaces,
        and initializes all the tracking variables needed for game state management,
        mapping, navigation, and reward calculation.
        
        Args:
            max_episodes (int): Maximum number of steps per episode before timeout (default: 500000)
            
        The initialization process:
        1. Sets up the Gymnasium spaces (observation and action)
        2. Initializes the PyBoy emulator with the Pokemon Yellow ROM
        3. Creates data structures for SLAM mapping and navigation
        4. Sets up variables for tracking game state and agent progress
        5. Initializes Pokemon party tracking
        6. Sets up the LLM controller for advanced decision making
        
        Key data structures:
        - curiosity_map: Tracks interest in exploring different map locations
        - frame_stack: Stores recent observations for temporal information
        - map: Stores the discovered game world (free spaces, obstacles)
        - transitions: Records locations that connect different maps
        - frontiers: Tracks unexplored edges of the known map
        - collision_counts: Tracks collisions at each position
        """
        super().__init__()
        # Initialize Gymnasium spaces for RL algorithms
        self.observation_space = Box(low=0, high=255, shape=(120, 120, 9), dtype=np.uint8)
        self.action_space = Discrete(6)
        
        # Initialize PyBoy emulator
        self.pyboy = PyBoy('ROMs/Pokemon_Yellow.gbc', window='null')
        self.pyboy_counter = 0
        
        # SLAM mapping variables
        self.map_size = 100  # Size of the internal map grid
        self.map_offset = self.map_size // 2  # Center offset for the map
        self.cell_size = 10  # Size of each cell in pixels for visualization
        self.map_image = np.ones(
            (self.map_size * self.cell_size, self.map_size * self.cell_size, 3), dtype=np.uint8
        ) * 255  # White background for map visualization
        self.map = defaultdict(lambda: defaultdict(int))  # 3D map structure: map_id -> (x,y) -> value
        self.map_offsets = {}  # Tracks offsets for each map id to align coordinates
        
        # Navigation variables
        self.curiosity_map = defaultdict(lambda: defaultdict(lambda: 1.0))  # High curiosity initially
        self.curiosity_decay_rate = 0.1  # Define how fast curiosity wanes with visits
        self.collision_counts = defaultdict(int)  # Track collisions at each position
        self.encounter_positions = defaultdict(set)  # Track where Pokemon encounters happen
        self.transitions = defaultdict(set)  # Track map transition locations
        self.transition_positions = defaultdict(set)  # map_id -> set of (x, y) transition positions
        self.frontiers = set()  # Set of unexplored frontiers
        self.current_frontier = None  # Current target frontier
        self.old_frontier = None  # Previous frontier for fallback
        self.current_path = []  # Current planned path
        self.visited_frontiers = set()  # Frontiers already visited
        self.excluded_frontiers = set()  # Frontiers to avoid
        self.fully_explored_maps = set()  # Maps that have been fully explored
        self.last_transition_index = -1  # Index for cycling through transitions
        
        # Game state tracking
        self.max_episodes = max_episodes  # Max steps before timeout
        self.agent_id = random.randint(1, 1000000)  # Unique ID for this agent instance
        self.first_episode = True  # Flag for first episode special handling
        self.initial_observation = True  # Flag for first observation
        self.current_map_id = None  # Current map ID from game memory
        self.ash_position = None  # Current position (map_id, x, y)
        self.ash_old_position = None  # Previous position for comparison
        
        # Observation processing
        self.frame_stack = deque(maxlen=3)  # Stack multiple frames for temporal information
        self.frame_skip = 1  # Number of frames to skip between actions
        self.frame_wait = 24  # Frames to wait in certain situations
        self.frame_wait_count = 0  # Counter for frame waiting
        
        # Performance metrics
        self.total_reward = 0  # Total accumulated reward
        self.episode_length = 0  # Current episode length
        self.current_score = 0  # Current game score
        self.level_progress = [-1]  # Track map progression
        self.level_progress_pct = 0.0  # Percentage of level progress
        
        # Pokemon tracking
        self.pokemon_found_list = [-1]  # IDs of Pokemon encountered
        self.pokemon_caught_list = [-1]  # IDs of Pokemon caught
        self.pokemon_tally = 0  # Count of Pokemon caught
        self.pkm_fnd = -1  # ID of most recently found Pokemon
        self.pkm_cau = -1  # ID of most recently caught Pokemon
        
        # Battle state tracking
        self.is_battling_fl = False  # Flag for battle state
        self.new_total_hp = 0  # Track player Pokemon HP
        self.new_enemy_hp = 0  # Track enemy Pokemon HP
        self.previous_enemy_lvl = 1000  # Previous enemy level
        self.ash_hp = 0  # Ash's current HP
        
        # Action tracking
        self.last_action = None  # Most recent action taken
        self.previous_action = None  # Action before last_action
        self.preferred_action = None  # Action to prefer in certain situations
        self.ppo_action = 0  # Action from PPO model
        
        # Stuck detection and recovery
        self.ash_stuck_counter = 0  # Counter for detecting when Ash is stuck
        self.ash_loc_history = deque(maxlen=50)  # History of recent positions
        self.steps_at_same_position = 0  # Steps spent at the same position
        self.steps_since_last_frontier = 0  # Steps since last frontier was reached
        self.steps_towards_current_frontier = 0  # Steps taken towards current frontier
        self.random_action_steps_remaining = 0  # Steps left for random action
        
        # Goal tracking
        self.global_goal = ''  # Current goal ('blue', 'red', 'magenta', 'green')
        self.goal_count = 0  # Counter for goal switching
        self.rand_count = 0  # Counter for randomness
        
        # Pokemon party tracking
        self.pkm_one = None  # First party Pokemon
        self.pkm_two = None  # Second party Pokemon
        self.pkm_three = None  # Third party Pokemon
        self.pkm_four = None  # Fourth party Pokemon
        self.pkm_five = None  # Fifth party Pokemon
        self.pkm_six = None  # Sixth party Pokemon
        self.current_box = None  # Current Pokemon box
        
        # Other state variables
        self.truncated = False  # Episode truncation flag
        self.first_loc_check = 0  # First location check flag
        
        # Initialize LLM controller for advanced decision making
        self.llm_controller = LLMController()
        
        print('STARTED AGENT: ', self.agent_id)
        self.reset_game_state()  # Initialize game state

    def reset_game_state(self):
        """
        Reset the game state variables to their initial values.
        
        This method is called during initialization and whenever the environment
        is reset between episodes. It resets all tracking variables, maps, and counters
        but does not reset the PyBoy emulator state itself (that's handled in reset()).
        
        The reset process:
        1. Clears performance metrics (rewards, scores, episode length)
        2. Resets Pokemon tracking lists and battle state flags
        3. Clears navigation data (maps, frontiers, paths)
        4. Resets all position tracking and mapping variables
        
        This method ensures that each new episode starts with a clean state while
        maintaining the PyBoy emulator instance for efficiency.
        
        Dependencies:
        - Called by __init__() during initial setup
        - Called by reset() at the start of each new episode
        - Required before reset_game_in_gui() which handles actual game state reset
        """
        # Reset performance metrics
        self.total_reward = 0
        self.episode_length = 0
        self.current_score = 0
        self.level_progress = [-1]
        self.pokemon_found_list = [-1]
        self.pokemon_caught_list = [-1]
        self.level_progress_pct = 0.0
        
        # Reset stuck detection
        self.ash_stuck_counter = 0
        self.ash_loc_history = deque(maxlen=50)
        self.steps_since_last_frontier = 0
        self.steps_at_same_position = 0
        self.random_action_steps_remaining = 0
        
        # Reset battle state
        self.is_battling_fl = False
        self.truncated = False
        self.new_total_hp = 0
        self.first_loc_check = 0
        self.new_enemy_hp = 0
        self.previous_enemy_lvl = 1000
        
        # Reset goals and actions
        self.global_goal = ''
        self.goal_count = 0
        self.rand_count = 0
        self.last_action = None
        self.previous_action = None
        self.preferred_action = None
        
        # Reset Pokemon tracking
        self.pkm_fnd = -1
        self.pkm_cau = -1
        self.pokemon_tally = 0
        
        # Reset navigation and mapping
        self.visited_frontiers = set()
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
        
        # Reset exploration tracking
        self.fully_explored_maps = set()
        self.transition_positions = defaultdict(set)
        self.last_transition_index = -1
        self.frame_wait_count = 0
        self.old_frontier = None
        
        # Reset Pokemon party and health
        self.ash_hp = 0
        self.pkm_one = None
        self.pkm_two = None
        self.pkm_three = None
        self.pkm_four = None
        self.pkm_five = None
        self.pkm_six = None
        self.current_box = None

    def step(self, action):
        """
        Execute one time step within the environment.
        
        This is the core method of the environment that processes agent actions,
        advances the game state, and calculates rewards. It handles both the direct
        execution of actions in the PyBoy emulator and the higher-level logic for
        game state tracking, map updates, and intelligent navigation.
        
        The step process:
        1. Updates the current goal based on game state
        2. Checks if the agent is in a battle
        3. Executes the action in the game with proper frame timing
        4. Updates Ash's position, map data, and path planning
        5. Calculates rewards based on the current goal and game events
        6. Returns the new observation, reward, done flag, and info dictionary
        
        A key feature is the goal-specific behavior:
        - In 'blue' (exploration) mode: Uses SLAM-based pathfinding (slam_action)
        - In other modes: Executes the agent's selected actions directly
        
        Args:
            action (int): The action to take (0-5, representing game controls)
            
        Returns:
            tuple: (observation, reward, done, truncated, info)
                - observation: The current game state as a stacked frame array
                - reward: The reward achieved by the previous action
                - done: Whether the episode has ended
                - truncated: Whether the episode was truncated (e.g., timeout)
                - info: Additional information (empty dictionary currently)
                
        Dependencies:
        - Calls get_goal() to determine the current objective
        - Calls battling() to check battle state
        - Calls execute_action() to send inputs to the game
        - Calls update_slam_map() to update the navigation map
        - Calls calculate_reward_and_done() for reward computation
        """
        reward = 0
        done = False
        info = {}
        
        # Process each frame in the frame skip window
        for _ in range(self.frame_skip):
            if not done:
                # Update current goal and check battle state
                self.get_goal()
                self.battling()
                self.random_action_steps_remaining = 0
                self.ppo_action = action  # Store the action from the RL agent
                
                # Track previous position for movement detection
                previous_ash_position = self.ash_position
                
                # Handle frame waiting period (used for game state stabilization)
                if self.frame_wait_count <= self.frame_wait:
                    # During wait period, send no action but tick the emulator
                    self.last_action = None
                    self.execute_action(self.last_action)
                    self.pyboy.tick(1, False)
                    
                    # Update position tracking if not in battle
                    if not self.is_battling_fl:
                        self.ash_old_position = self.ash_position
                        self.ash_position = self.get_ash_position()
                        self.ash_loc_history.append(self.ash_position)
                        self.update_slam_map()

                    # Update observation frame stack
                    self.update_frame_stack()

                    # Progress episode and calculate rewards
                    self.episode_length += 1
                    reward, done = self.calculate_reward_and_done(self.last_action)
                    truncated = self.truncated
                    
                    # Handle truncation
                    if truncated:
                        info = {}
                        
                    self.frame_wait_count += 1
                    return self.get_stacked_observation(), reward, done, truncated, info
                
                # Reset frame wait counter for normal operation
                self.frame_wait_count = 0
                
                # Handle different goal modes
                if self.global_goal == 'blue' and not self.is_battling_fl:
                    # Exploration mode - use SLAM-based pathfinding
                    action = self.slam_action()
                    self.last_action = action
                    self.execute_action(self.last_action)
                    self.pyboy.tick(1, False)
                    
                    # Update position for map building
                    self.ash_old_position = self.ash_position
                    self.ash_position = self.get_ash_position()
                    self.ash_loc_history.append(self.ash_position)
                    self.update_slam_map()
                else:
                    # Other modes - execute agent's action directly
                    self.last_action = action
                    self.execute_action(self.last_action)
                    self.pyboy.tick(1, False)
                    
                    # Update position if not in battle
                    if not self.is_battling_fl:
                        self.ash_old_position = self.ash_position
                        self.ash_position = self.get_ash_position()
                        self.ash_loc_history.append(self.ash_position)
                        self.update_slam_map()

                # Track if Ash is stuck at the same position
                if not self.is_battling_fl:
                    if self.ash_position == previous_ash_position:
                        self.steps_at_same_position += 1
                    else:
                        self.steps_at_same_position = 0
                else:
                    self.steps_at_same_position = 0

                # Update observation and visualization
                self.update_frame_stack()
                self.render()
                self.render_map()
                
                # Progress episode and calculate rewards
                self.episode_length += 1
                reward, done = self.calculate_reward_and_done(self.last_action)
                truncated = self.truncated
                
                # Handle truncation
                if truncated:
                    info = {}

        return self.get_stacked_observation(), reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state and return the initial observation.
        
        This method handles the full reset process for the environment, including:
        1. Periodically recreating the PyBoy emulator to prevent memory issues
        2. Resetting internal state variables via reset_game_state()
        3. Loading a saved game state to ensure consistent starting conditions
        4. Initializing the map with Ash's current position
        5. Setting up the observation frame stack
        
        The reset ensures that each new episode starts with a clean, consistent
        game state while maintaining the necessary components for continued learning.
        
        Args:
            seed (Optional[int]): Random seed for reproducibility
            options (Optional[dict]): Additional options for reset (not currently used)
            
        Returns:
            tuple: (observation, info)
                - observation: Initial stacked observation of the reset environment
                - info: Empty dictionary (reserved for future metadata)
                
        Dependencies:
        - Calls reset_game_state() to reset internal variables
        - Calls reset_game_in_gui() to reload the game save state
        - Calls get_map_id() and get_ash_position() to initialize map positioning
        - Calls initialize_map_offset() to set up map coordinate system
        - Calls detect_frontiers() to find exploration targets
        """
        # Periodically recreate the PyBoy emulator to prevent memory leaks
        if self.pyboy_counter == 10000:
            self.pyboy.stop()
            del self.pyboy
            self.pyboy_counter = 0
            self.pyboy = PyBoy('ROMs/Pokemon_Yellow.gbc', window_type="headless", scale=3, game_wrapper=False)
        
        # Set random seed if provided for reproducibility
        if seed:
            np.random.seed(seed)
        
        # Reset internal state variables
        self.reset_game_state()
        
        # Reset the game by loading a saved state
        self.reset_game_in_gui()
        
        # Initialize observation frame stack if this is the first observation
        if self.initial_observation:
            self.get_goal()
            l_obs = self.get_observation()
            self.frame_stack.extend([l_obs] * self.frame_stack.maxlen)
        
        # Increment counter for tracking PyBoy instance age
        self.pyboy_counter += 1
        
        # Get current map and position information
        self.current_map_id = self.get_map_id()
        self.ash_position = self.get_ash_position()
        
        # Initialize map coordinate system based on current position
        self.initialize_map_offset(self.current_map_id, self.ash_position[1], self.ash_position[2])
        
        # Mark current position as explored (value 1)
        self.add_to_map(self.ash_position, 1)
        
        # Find initial exploration frontiers
        self.detect_frontiers()
        
        # Get initial Pokemon party information
        self.pkm_one = self.pyboy.memory[0xD16A]
        self.pkm_two = self.pyboy.memory[0xD196]
        self.pkm_three = self.pyboy.memory[0xD1C2]
        self.pkm_four = self.pyboy.memory[0xD1EE]
        self.pkm_five = self.pyboy.memory[0xD21A]
        self.pkm_six = self.pyboy.memory[0xD246]
        self.current_box = self.pyboy.memory[0xDA7F]
        
        return self.get_stacked_observation(), {}

    def initialize_map_offset(self, map_id, x, y):
        """
        Initialize the offset for a map's coordinate system.
        
        This method establishes a mapping between game world coordinates and
        the internal grid coordinate system. When a new map is encountered, 
        it calculates offsets so that:
        1. The agent's current position is centered in the internal map grid
        2. Consistent grid references can be maintained across map transitions
        3. The world can be represented in a fixed-size grid despite having many areas
        
        Map offsets are crucial for the SLAM (Simultaneous Localization and Mapping)
        system to create a consistent map of the game world.
        
        Args:
            map_id (int): The identifier for the current map area
            x (int): The agent's x-coordinate in the game world
            y (int): The agent's y-coordinate in the game world
            
        Returns:
            None - Updates self.map_offsets dictionary with the new offset values
            
        Potential improvements:
        - Add validation for map_id and coordinate values
        - Handle special cases for map transitions more intelligently
        - Consider dynamic resizing for maps that exceed the grid size
        """
        if map_id not in self.map_offsets:
            offset_x = self.map_size // 2 - x
            offset_y = self.map_size // 2 - y
            self.map_offsets[map_id] = (offset_x, offset_y)
            logger.debug(f"Initialized map offset for Map ID {map_id}: Offset X = {offset_x}, Offset Y = {offset_y}")

    def render(self):
        """
        Render the game screen with goal indicator for visualization.
        
        This method processes and displays the current game screen with additional
        visual elements to help understand the agent's state and goals:
        1. Captures the raw screen from the PyBoy emulator
        2. Converts it from BGR to RGB color format
        3. Handles cases where screen data is unavailable
        4. Resizes the screen to a standard 360x360 size for display
        5. Adds a color block indicator representing the current goal
        6. Also calls render_map() to show a top-down view of the explored environment
        
        The render method is essential for:
        - Debugging agent behavior
        - Visualizing the agent's progress
        - Understanding the current state of the game
        - Monitoring the exploration process
        
        Args:
            None
            
        Returns:
            None - Side effect is displaying windows with the game screen and map
            
        Potential improvements:
        - Add optional parameters to control visualization details
        - Implement recording capability for creating videos
        - Add more visual indicators of agent state
        - Create an option for headless operation when visualization isn't needed
        """
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
        """
        Render a detailed SLAM (Simultaneous Localization and Mapping) visualization of the game world.
        
        This method generates a comprehensive visualization that displays:
        1. The current map grid with explored and unexplored areas
        2. Obstacles and free spaces in different colors
        3. Pokemon encounter locations
        4. Map transitions (doors, stairs, etc.)
        5. Frontier areas (edges of exploration)
        6. The current target frontier location
        7. The planned path for navigation
        8. Ash's current position
        9. Game state information (current goal, score, last action)
        10. Pokemon party information with sprites
        
        The visualization uses color-coding for different elements:
        - Black: Free/walkable spaces
        - Green: Obstacles/walls
        - Yellow: Pokemon encounter locations
        - Brown: Map transitions (doors, entrances)
        - Blue: Frontier areas (unexplored boundaries)
        - Orange: Current target frontier
        - Purple: Planned path
        - Red with Blue border: Ash's current position
        
        The method also displays text overlays with game state information and
        Pokemon party details, including sprites of captured Pokemon.
        
        Args:
            None
            
        Returns:
            None (displays the map via OpenCV window)
            
        Potential improvements:
        - Add configuration options for display elements
        - Implement zoom and pan functionality for large maps
        - Add a mini-map for overall context
        - Provide a toggle for different visualization modes
        - Add a heatmap overlay for encounter frequencies
        """
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

        # Render the global goal in the top right corner
        text_position = (self.map_image.shape[1] - 1000, 30)  # Adjusted position
        cv2.putText(self.map_image, f"Goal: {self.global_goal}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Render the last action in the top right corner
        text_position_action = (self.map_image.shape[1] - 1000, 60)  # Adjusted position
        cv2.putText(self.map_image, f"Last Action: {self.last_action}", text_position_action, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Render the total Pokemon levels in the top right corner
        text_position_score = (self.map_image.shape[1] - 1000, 90)  # Adjusted position
        cv2.putText(self.map_image, f"Total Pokemon Levels: {self.current_score}", text_position_score, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Render the Pokemon 1 id
        if self.pkm_one:
            pokemon_info = self.get_pokemon_sprite(self.pkm_one)

            pokemon_image = cv2.imread(f"sprites/sprites/pokemon/versions/generation-i/yellow/{pokemon_info['ID']}.png")
            pokemon_image = cv2.resize(pokemon_image, (25, 25))
            # Define the position where you want to place the image
            x_offset, y_offset = self.map_image.shape[1] - 640, 10
            # Overlay the image onto the map_image
            self.map_image[y_offset:y_offset + pokemon_image.shape[0], x_offset:x_offset + pokemon_image.shape[1]] = pokemon_image  
        text_position_score = (self.map_image.shape[1] - 790, 30)  # Adjusted position
        cv2.putText(self.map_image, f"Pokemon Party 1: ", text_position_score, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        

        # Render the Pokemon 2 id
        if self.pkm_two:
            pokemon_info = self.get_pokemon_sprite(self.pkm_two)

            pokemon_image = cv2.imread(f"sprites/sprites/pokemon/versions/generation-i/yellow/{pokemon_info['ID']}.png")
            pokemon_image = cv2.resize(pokemon_image, (25, 25))
            # Define the position where you want to place the image
            x_offset, y_offset = self.map_image.shape[1] - 640, 40
            # Overlay the image onto the map_image
            self.map_image[y_offset:y_offset + pokemon_image.shape[0], x_offset:x_offset + pokemon_image.shape[1]] = pokemon_image  
        text_position_score = (self.map_image.shape[1] - 790, 60)  # Adjusted position
        cv2.putText(self.map_image, f"Pokemon Party 2: ", text_position_score, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Render the Pokemon 3 id
        if self.pkm_three:
            pokemon_info = self.get_pokemon_sprite(self.pkm_three)

            pokemon_image = cv2.imread(f"sprites/sprites/pokemon/versions/generation-i/yellow/{pokemon_info['ID']}.png")
            pokemon_image = cv2.resize(pokemon_image, (25, 25))
            # Define the position where you want to place the image
            x_offset, y_offset = self.map_image.shape[1] - 640, 70
            # Overlay the image onto the map_image
            self.map_image[y_offset:y_offset + pokemon_image.shape[0], x_offset:x_offset + pokemon_image.shape[1]] = pokemon_image  
        text_position_score = (self.map_image.shape[1] - 790, 90)  # Adjusted position
        cv2.putText(self.map_image, f"Pokemon Party 3: ", text_position_score, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Render the Pokemon 4 id
        if self.pkm_four:
            pokemon_info = self.get_pokemon_sprite(self.pkm_four)

            pokemon_image = cv2.imread(f"sprites/sprites/pokemon/versions/generation-i/yellow/{pokemon_info['ID']}.png")
            pokemon_image = cv2.resize(pokemon_image, (25, 25))
            # Define the position where you want to place the image
            x_offset, y_offset = self.map_image.shape[1] - 450, 10
            # Overlay the image onto the map_image
            self.map_image[y_offset:y_offset + pokemon_image.shape[0], x_offset:x_offset + pokemon_image.shape[1]] = pokemon_image  
        text_position_score = (self.map_image.shape[1] - 600, 30)  # Adjusted position
        cv2.putText(self.map_image, f"Pokemon Party 4: ", text_position_score, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Render the Pokemon 5 id
        if self.pkm_five:
            pokemon_info = self.get_pokemon_sprite(self.pkm_five)

            pokemon_image = cv2.imread(f"sprites/sprites/pokemon/versions/generation-i/yellow/{pokemon_info['ID']}.png")
            pokemon_image = cv2.resize(pokemon_image, (25, 25))
            # Define the position where you want to place the image
            x_offset, y_offset = self.map_image.shape[1] - 450, 40
            # Overlay the image onto the map_image
            self.map_image[y_offset:y_offset + pokemon_image.shape[0], x_offset:x_offset + pokemon_image.shape[1]] = pokemon_image  
        text_position_score = (self.map_image.shape[1] - 600, 60)  # Adjusted position
        cv2.putText(self.map_image, f"Pokemon Party 5: ", text_position_score, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Render the Pokemon 6 id
        if self.pkm_six:
            pokemon_info = self.get_pokemon_sprite(self.pkm_six)

            pokemon_image = cv2.imread(f"sprites/sprites/pokemon/versions/generation-i/yellow/{pokemon_info['ID']}.png")
            pokemon_image = cv2.resize(pokemon_image, (25, 25))
            # Define the position where you want to place the image
            x_offset, y_offset = self.map_image.shape[1] - 450, 70
            # Overlay the image onto the map_image
            self.map_image[y_offset:y_offset + pokemon_image.shape[0], x_offset:x_offset + pokemon_image.shape[1]] = pokemon_image  
        text_position_score = (self.map_image.shape[1] - 600, 90)  # Adjusted position
        cv2.putText(self.map_image, f"Pokemon Party 6: ", text_position_score, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Render the total Pokemon in the current box transferred to PC
        text_position_score = (self.map_image.shape[1] - 400, 30)  # Adjusted position
        cv2.putText(self.map_image, f"Total in Box: {self.current_box}", text_position_score, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Render the current map id
        text_position_score = (self.map_image.shape[1] - 400, 60)  # Adjusted position
        cv2.putText(self.map_image, f"Current Map: {self.current_map_id}", text_position_score, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Render the current position 
        text_position_score = (self.map_image.shape[1] - 400, 90)  # Adjusted position
        cv2.putText(self.map_image, f"Current Target Frontier: {self.current_frontier}", text_position_score, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Display the map with the path
        cv2.imshow("SLAM Map", self.map_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def close(self):
        """
        Close the environment and release resources.
        
        This method handles cleanup by:
        1. Closing any open OpenCV windows using cv2.destroyAllWindows()
        2. Stopping the PyBoy emulator instance
        
        The method is called when:
        - The environment is explicitly closed by the user
        - The 'q' key is pressed during map rendering
        - The Python process is terminated
        
        Args:
            None
            
        Returns:
            None
            
        Potential improvements:
        - Add additional cleanup for any other resources (e.g., logging handlers)
        - Add graceful handling of exceptions during shutdown
        - Consider adding a method to save the game state before closing
        """
        cv2.destroyAllWindows()
        self.pyboy.stop()

    def get_observation(self):
        """
        Get the current game state as an image observation.
        
        This method:
        1. Sets Poké Ball count to 99 in game memory (address 0xd31e)
        2. Retrieves the current goal color
        3. Captures the raw screen data from the PyBoy emulator
        4. Converts the color format from BGR to RGB
        5. Handles the case where screen data is unavailable by returning a blank image
        6. Resizes the image to 120x120 pixels for standardized input to neural networks
        7. Adds a colored block to indicate the current goal
        
        Args:
            None
            
        Returns:
            np.ndarray: A processed RGB image (120x120x3) representing the current game state with a goal indicator
            
        Potential improvements:
        - Add error handling for emulator state issues
        - Consider normalizing pixel values (to range 0-1) for neural network input
        - Add optional preprocessing like grayscale conversion or frame differencing
        - Comment the purpose of setting Poké Ball count to 99
        """
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
        """
        Get a stacked observation by concatenating multiple recent frames.
        
        This method creates a temporal representation of the game state by combining
        multiple consecutive frames stored in the frame_stack deque. Frame stacking
        is a common technique in reinforcement learning to provide temporal context,
        allowing the agent to detect motion and changes over time.
        
        The method simply concatenates all frames in the stack along the channel axis,
        resulting in an observation with shape (120, 120, C*3) where C is the number
        of frames in the stack.
        
        Args:
            None
            
        Returns:
            np.ndarray: Concatenated frames with shape (120, 120, C*3) where C is
                        the number of frames in self.frame_stack
                        
        Potential improvements:
        - Add checks to ensure frame_stack is not empty before concatenating
        - Consider different stacking methods (e.g., max pooling across frames)
        - Add support for different observation types beyond image data
        - Consider adding channels-first option for frameworks like PyTorch
        """
        return np.concatenate(self.frame_stack, axis=-1)

    def execute_action(self, action):
        """
        Execute a specified action in the game environment.
        
        This method translates the abstract action integer into a concrete key press
        in the PyBoy emulator. The method follows these steps:
        1. Releases any previously pressed keys to ensure clean input
        2. Maps the action integer to the corresponding PyBoy button name
        3. Sends the input event to the emulator
        4. Applies special logic for movement actions to handle direction changes
        5. Tracks the previous action for movement continuity
        
        The action mapping is:
        - 1: Move right
        - 2: Move left
        - 3: Move up
        - 4: Move down
        - 0: Interact/select (A button)
        - 5: Cancel/run (B button)
        
        Args:
            action (int): The action to execute, corresponding to the mapping above
            
        Returns:
            None - Side effects are within the emulator state
            
        Potential improvements:
        - Add input validation and error handling for invalid actions
        - Consider mapping more complex action combinations (e.g., running while moving)
        - Add debug logging to track action execution more clearly
        - Handle edge cases like menu navigation vs. overworld movement differently
        """
        self.release_all_keys()
        button_map = {
            1: "right",
            2: "left",
            3: "up",
            4: "down",
            0: "a",
            5: "b",
        }
        button = button_map.get(action)
        
        if button:
            ticks_needed = 1
            if action in [1, 2, 3, 4] and not self.is_battling_fl:
                if self.previous_action is not None:
                    opposite_actions = {1: 2, 2: 1, 3: 4, 4: 3}
                    if action == opposite_actions.get(self.previous_action):
                        ticks_needed = 2
            
            # Using button method which automatically releases on next tick
            self.pyboy.button(button)
            logger.debug(f"Executing action {action}: Sent {button}")
            
            for _ in range(ticks_needed):
                self.pyboy.tick(1, True)
                
            if action in [1, 2, 3, 4]:
                self.previous_action = action
                logger.debug(f"Updated previous_action to {self.previous_action}")

    def update_frame_stack(self):
        """
        Update the frame stack with the latest observation.
        
        This method:
        1. Gets the current observation using get_observation()
        2. Adds it to the frame_stack deque
        3. When the deque is full, it automatically removes the oldest frame
        
        The frame stack is used to provide temporal context to the agent,
        allowing it to perceive motion and changes across multiple frames
        rather than just static images.
        
        Args:
            None
            
        Returns:
            None - Updates the frame_stack data structure internally
            
        Potential improvements:
        - Add initialization check to ensure frame_stack exists
        - Handle special initialization case when frame_stack is empty
        - Add option to skip frames to reduce computation
        - Consider more memory-efficient frame representations
        """
        observation = self.get_observation()
        self.frame_stack.append(observation)

    def calculate_reward_and_done(self, action):
        """
        Calculate rewards and determine if the episode should end.
        
        This method serves as the main reward processing function that:
        1. Gets the basic reward by calling calculate_reward_basic()
        2. Accumulates rewards in the total_reward tracker
        3. Determines episode termination based on truncation flag
        4. Provides debugging output about episode performance when done
        
        The reward function is critical for reinforcement learning as it:
        - Shapes the agent's behavior toward desired goals
        - Signals when the episode should terminate
        - Provides feedback on the agent's performance
        - Tracks cumulative rewards across the episode
        
        Args:
            action (int): The last action taken by the agent
            
        Returns:
            tuple: (reward, done)
                - reward (float): The calculated reward value
                - done (bool): Flag indicating if the episode should terminate
                
        Potential improvements:
        - Add more detailed information dictionary for debugging
        - Implement reward normalization or scaling
        - Add early termination conditions based on performance
        - Track reward components separately for better analysis
        """
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
        """
        Calculate the basic reward signal based on the agent's actions and game state.
        
        This method implements the core reward function that reinforces:
        1. Exploration (for 'blue' goal) - rewards for discovering new areas
        2. Battle success (for 'red' goal) - rewards for battling and defeating Pokemon
        3. Pokemon collection (for 'magenta' goal) - rewards for finding and catching Pokemon
        4. Penalties for getting stuck or timing out
        
        The reward structure is goal-dependent, with different behaviors reinforced
        based on the current global_goal value. This method handles:
        - Map exploration rewards
        - Battle detection and rewards
        - Pokemon encounter rewards
        - Leveling up rewards
        - Stuck penalties
        
        Args:
            action (int): The action that was taken by the agent
            
        Returns:
            float: The calculated reward value
            
        Potential improvements:
        - Refactor into smaller, goal-specific reward functions
        - Add more granular reward signals for complex behaviors
        - Implement a configurable reward weighting system
        - Add reward shaping to provide smoother learning gradients
        """
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
        """
        Detect if the player's Pokemon HP has decreased since the last check.
        
        This method monitors the player's Pokemon team health by:
        1. Reading HP values from six memory addresses (one for each party slot)
        2. Comparing the current total HP to the previously recorded value
        3. Updating the tracking variable if a drop is detected
        
        HP tracking is critical for:
        - Battle performance evaluation in reward calculations
        - Detecting when the player is taking damage
        - Informing battle strategy decisions
        
        Args:
            None
            
        Returns:
            bool: True if HP has dropped since last check, False otherwise
            
        Dependencies:
        - Reads memory addresses 0xD16B, 0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C
        - Updates self.new_total_hp to track HP between calls
        - Used by battle-related reward calculations
        """
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
        """
        Detect if the player has dealt damage to an enemy Pokemon.
        
        This method tracks enemy Pokemon HP during battles by:
        1. Checking if a battle is currently active
        2. Reading the enemy Pokemon's current HP from memory
        3. Comparing it to the previously recorded enemy HP value
        4. Updating the tracking variable if damage was dealt
        
        Damage detection is used for:
        - Battle performance evaluation in reward calculations
        - Tracking battle progress and effectiveness
        - Informing battle strategy decisions
        
        Args:
            None
            
        Returns:
            bool: True if damage was dealt to enemy since last check, False otherwise
            
        Dependencies:
        - Requires self.is_battling_fl to be properly set by battling() method
        - Reads memory address 0xcfe6 for enemy HP
        - Updates self.new_enemy_hp to track enemy HP between calls
        - Only functions during active battles (returns False otherwise)
        """
        damaged = False
        if self.is_battling_fl:
            enemy_hp = self.pyboy.memory[0xcfe6]
            if enemy_hp < self.new_enemy_hp:
                damaged = True
                self.new_enemy_hp = enemy_hp
                return damaged
        return damaged

    def reset_game_in_gui(self):
        """
        Reset the game to a saved state for a new episode.
        
        This method handles the game state reset process by:
        1. Advancing the emulator by one tick to ensure stable state
        2. Selecting the appropriate save state file based on episode count
        3. Verifying the save state file exists
        4. Loading the save state into the PyBoy emulator
        
        The reset process ensures:
        - Consistent starting conditions for each episode
        - Proper initialization of the game state
        - Reliable environment behavior for reinforcement learning
        
        Args:
            None
            
        Returns:
            None
            
        Raises:
            FileNotFoundError: If the specified save state file doesn't exist
            
        Dependencies:
        - Requires PyBoy emulator to be properly initialized
        - Uses self.first_episode flag to determine which save to load
        - Depends on save state files in the ROMs directory
        - Called by reset() method at the start of each episode
        """
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
        """
        Check the number of Pokemon caught by the player.
        
        This method reads specific memory addresses in the game to determine
        the total number of Pokemon caught. It combines values from two memory
        locations that track different aspects of Pokemon collection.
        
        The method is used for:
        1. Reward calculation in catch-focused goal modes
        2. Tracking progress through the game
        3. Determining when to switch between different goal modes
        
        Args:
            None
            
        Returns:
            int: The combined value representing Pokemon caught
            
        Dependencies:
        - Reads memory addresses 0xd162 and 0xda7f from the PyBoy emulator
        - Used by reward calculation functions to determine catch-based rewards
        """
        v_1 = self.pyboy.memory[0xd162]
        v_2 = self.pyboy.memory[0xda7f]
        values = v_1 + v_2
        return values

    def get_score(self):
        """
        Calculate the player's score based on game progress.
        
        This method tracks the player's score by:
        1. Reading values from six memory addresses related to game progress
        2. Comparing the current sum to the previously recorded score
        3. Calculating a reward based on the difference (if any)
        4. Updating the current_score tracking variable
        
        The score calculation includes:
        - Detecting when new progress has been made
        - Scaling rewards by multiplying by 10 for significant progress
        - Handling the initial case when current_score is 0
        
        Args:
            None
            
        Returns:
            int: The calculated score/reward value
                - 0 if no progress was made
                - (new_total - previous_total) * 10 if progress was made
                
        Dependencies:
        - Reads memory addresses 0xd18b, 0xd1b7, 0xd1e3, 0xd20f, 0xd23b, 0xd267
        - Updates self.current_score to track progress between calls
        - Used by reward calculation functions to determine progress-based rewards
        """
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
        """
        Check if the agent is currently in a battle state.
        
        This method determines if a battle is active by:
        1. Reading player Pokemon HP values from memory addresses
        2. Updating the ash_hp tracking variable with the current total HP
        3. Checking the battle state flag at memory address 0xd056
        
        This is a quick check used by various systems including:
        - The reward calculation to determine battle-specific rewards
        - The collision and movement systems to adjust behavior during battles
        - The stuck detection system to avoid false positives during battles
        
        Args:
            None
            
        Returns:
            bool: True if currently in a battle, False otherwise
            
        Potential improvements:
        - Add more robust battle state detection using multiple indicators
        - Track battle type (wild Pokemon, trainer, etc.)
        - Cache results to reduce memory reads
        - Add detailed battle state information for debugging
        """
        player_hp_addresses = [0xD16B, 0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C]
        total_hp = sum([self.pyboy.memory[address] for address in player_hp_addresses])
        self.ash_hp = total_hp
        return self.pyboy.memory[0xd056] != 0

    def battling(self):
        """
        Detect and handle Pokemon battles, updating battle state and tracking variables.
        
        This method is responsible for:
        1. Detecting when a battle begins or ends by checking game memory
        2. Tracking player Pokemon HP and party information during battles
        3. Tracking enemy Pokemon information (HP, level)
        4. Recording encounter positions on the map for future reference
        
        The battle detection is a critical component of the environment as it:
        - Triggers different reward calculations based on battle outcomes
        - Affects the goal selection logic in get_goal()
        - Determines when to use battle-specific actions vs. navigation actions
        - Helps track Pokemon encounters for mapping and statistics
        
        Args:
            None
            
        Returns:
            bool or None: 
                - True if a battle just started
                - False if a battle just ended
                - None if battle state didn't change
                
        Side effects:
            - Updates self.is_battling_fl flag
            - Updates Pokemon party information
            - Updates HP tracking variables
            - Adds encounter positions to the map
            
        Dependencies:
        - Reads various memory addresses from the PyBoy emulator
        - Uses get_ash_position_grid() to get the current position
        - Updates encounter_positions map with battle locations
        """
        battle_state = self.pyboy.memory[0xd056]
        if battle_state != 0:
            player_hp_addresses = [0xD16B, 0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C]
            total_hp = sum([self.pyboy.memory[address] for address in player_hp_addresses])
            self.pkm_one = self.pyboy.memory[0xD16A]
            self.pkm_two = self.pyboy.memory[0xD196]
            self.pkm_three = self.pyboy.memory[0xD1C2]
            self.pkm_four = self.pyboy.memory[0xD1EE]
            self.pkm_five = self.pyboy.memory[0xD21A]
            self.pkm_six = self.pyboy.memory[0xD246]
            self.current_box = self.pyboy.memory[0xDA7F]
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

    def is_transition_position(self, map_id: int, x: int, y: int) -> bool:
        """
        Check if a position is a known map transition point.
        
        This method determines if a specific position on a given map is a transition
        point that leads to another map. Transition positions are critical for:
        1. Building a complete map of the game world
        2. Enabling navigation between different map areas
        3. Identifying potential exploration targets
        
        The method is used during:
        - Path planning to handle transitions between maps
        - Frontier detection to identify map boundaries
        - Navigation to determine when map transitions occur
        
        Args:
            map_id (int): The identifier for the current map area
            x (int): The x-coordinate to check
            y (int): The y-coordinate to check
            
        Returns:
            bool: True if the position is a known transition point, False otherwise
            
        Dependencies:
        - Relies on self.transition_positions dictionary being properly maintained
        - Used by navigation and path planning systems
        """
        return (x, y) in self.transition_positions.get(map_id, set())

    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get 4-connected neighbors of a given position.
        
        This method returns the four cardinal neighbors (up, down, left, right)
        of a given position in the grid coordinate system. It:
        1. Calculates the coordinates of the four adjacent cells
        2. Filters out any coordinates that would be outside the valid map range
        
        The neighbors are used for:
        - Path planning and navigation
        - Frontier detection
        - Map building and exploration
        
        Args:
            position (Tuple[int, int]): The (x, y) position to get neighbors for
            
        Returns:
            List[Tuple[int, int]]: List of valid neighboring positions
            
        Dependencies:
        - Used by path planning algorithms
        - Used by frontier detection
        - Respects map size boundaries
        """
        x, y = position
        neighbors = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                neighbors.append((nx, ny))
        return neighbors

    def check_target_frontier_reached(self):
        """
        Check if the agent has reached the current target frontier position.
        
        This method is called regularly to determine if the agent has successfully
        reached its current exploration target. When a frontier is reached, it:
        1. Handles any map transitions if the frontier is a transition point
        2. Marks the frontier as visited to prevent revisiting
        3. Resets path planning variables for the next exploration target
        
        The frontier system is central to the exploration strategy, allowing the agent to:
        - Systematically explore unknown areas of the game world
        - Build a comprehensive map of the environment
        - Navigate efficiently between known and unknown areas
        
        Args:
            None
            
        Returns:
            None - Updates internal state variables
            
        Side effects:
        - May update self.visited_frontiers and self.current_frontier
        - Resets path planning variables when a frontier is reached
        - May trigger map transition handling
        
        Dependencies:
        - Calls get_ash_position_grid() to determine current position
        - Calls is_transition_position() to check for map transitions
        - Calls handle_map_transition() when at transition points
        - Updates exploration tracking variables
        """
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
        """
        Process a map transition point when the agent reaches one.
        
        This method is called when the agent reaches a position that may lead
        to another map area. It performs several critical functions:
        1. Validates the transition point by checking neighboring positions
        2. Removes isolated transition points that may be false positives
        3. Records confirmed transition points for future navigation
        4. Updates tracking sets to manage exploration targets
        
        Map transitions are crucial for:
        - Building a complete map of the game world
        - Enabling navigation between different areas
        - Preventing the agent from getting stuck at map boundaries
        
        Args:
            map_id (int): The identifier for the current map area
            x (int): The x-coordinate of the transition position
            y (int): The y-coordinate of the transition position
            
        Returns:
            None - Updates internal state variables
            
        Side effects:
        - May update self.transition_positions dictionary
        - May update self.excluded_frontiers and self.visited_frontiers sets
        - Logs information about transition handling
        
        Dependencies:
        - Calls get_neighbors() to check surrounding positions
        - Uses self.map to verify if neighbors are explored
        - Only records valid transitions when agent has non-zero HP
        """
        logger.info(f"Map transition reached at ({map_id}, {x}, {y}). Handling transition.")
        # Check for isolated transition positions and remove them
        neighbors = self.get_neighbors((x, y))
        is_isolated = True
        for nx, ny in neighbors:
            if self.map[map_id].get((nx, ny), 0) != 0:  # If any neighbor is explored
                is_isolated = False
        
        if is_isolated:
            logger.warning(f"Isolated transition position detected at ({map_id}, {x}, {y}). Removing it.")
            self.transition_positions[map_id].discard((x, y))
            return  # Skip adding this transition position
        
        # If not isolated, proceed with adding the transition position  
        if self.ash_hp != 0:
            self.transition_positions[map_id].add((x, y))
            self.excluded_frontiers.add((map_id, x, y))
            self.visited_frontiers.add((map_id, x, y))
    
    def slam_action(self) -> int:
        """
        Determine the next action based on SLAM (Simultaneous Localization and Mapping) pathfinding.
        
        This method is the core navigation intelligence of the agent, responsible for:
        1. Following existing paths to target frontiers
        2. Detecting and planning paths to new frontiers when needed
        3. Handling cases where no frontiers are available
        4. Integrating with the LLM controller for advanced decision making
        
        The method implements a hierarchical decision process:
        - If a path exists, follow it to the target frontier
        - If no path exists but frontiers are available, plan a new path
        - If no frontiers are available, try to move to unexplored maps
        - If no unexplored maps, use LLM controller for intelligent decisions
        - As a last resort, fall back to random actions
        
        Args:
            None
            
        Returns:
            int: The action to take (0-5, representing game controls)
            
        Dependencies:
        - Uses plan_path() for pathfinding
        - Uses detect_frontiers() to find exploration targets
        - Uses get_action_from_positions() to convert path steps to actions
        - Uses move_to_unexplored_map() to find transitions to new areas
        - Uses LLM controller for advanced decision making
        """
        # If we have a current path, follow it
        if self.current_path:
            next_position = self.current_path[0]
            current_position = self.get_ash_position_grid()

            # If we've reached the next position, advance to the next step in the path
            if current_position == next_position:
                self.current_path.pop(0)
                logger.info(f"Ash moved to {next_position}. Progressing path.")

                # Check if there are more steps in the path
                if self.current_path:
                    next_position = self.current_path[0]
                else:
                    # Path completed, switch to random action
                    logger.critical("Ash has reached the final destination of the current path.")
                    self.random_action_steps_remaining = 1
                    return self.ppo_action

            # Convert path positions to game action
            action = self.get_action_from_positions(current_position, next_position)
            logger.debug(f"Following path. Next action: {action} to move from {current_position} to {next_position}.")
            return action
        else:
            # No current path, detect frontiers for exploration
            self.detect_frontiers()
            
            # If we have a frontier target, plan a path to it
            if self.current_frontier:
                logging.critical(f"current_frontier = {self.current_frontier}") 
                logging.critical(f"frontiers = {self.frontiers}")
                
                # Plan path to the frontier
                path = self.plan_path(self.get_ash_position_grid(), self.current_frontier)
                
                # If no valid path exists, exclude this frontier and try another
                if path is None or len(path) < 2:
                    logger.critical("Cannot find a path to the frontier. Excluding the frontier.")
                    self.excluded_frontiers.add(self.current_frontier)
                    self.frontiers.discard(self.current_frontier)
                    if self.current_frontier:
                        self.old_frontier = self.current_frontier
                    self.current_frontier = None
                    self.goal_count = 500000  # Trigger goal switching
                    self.random_action_steps_remaining = 1
                    return self.ppo_action
                else:
                    # Valid path found, follow it
                    self.current_path = path[1:]  # Skip the current position
                    logger.info(f"Planned path to frontier: {self.current_path}")
                    current_position = self.get_ash_position_grid()
                    next_position = self.current_path[0]
                    action = self.get_action_from_positions(current_position, next_position)
                    return action
            else:
                # No frontiers available, try alternative strategies
                logger.critical(f"No current frontier, moving towards an unexplored map or fallback actions")

                # Try to move to an unexplored map via transitions
                moved = self.move_to_unexplored_map()
                if moved:
                    logger.critical("unexplored maps or transitions found. Returning path action.")
                    logger.info(f"Planned path to frontier: {self.current_path}")
                    current_position = self.get_ash_position_grid()
                    next_position = self.current_path[0]
                    action = self.get_action_from_positions(current_position, next_position)
                    return action

                # Get the current observation and game stats for LLM
                observation = self.get_observation()
                game_stats = self.llm_controller.get_game_stats(self)
                
                # Try to get action from LLM controller
                try:
                    llm_action = asyncio.get_event_loop().run_until_complete(
                        self.llm_controller.get_next_action(observation, game_stats)
                    )
                    logger.critical(f"llm_action = {llm_action}")
                    if llm_action is not None:
                        logger.critical(f"Using LLM suggested action: {llm_action}")
                        return llm_action
                except Exception as e:
                    logger.critical(f"Error getting LLM action: {e}")
                    return self.ppo_action  # Fallback to PPO action if LLM fails

                # If LLM fails, try moving towards known areas with high curiosity
                path = self.move_towards_known_area()
                if path:
                    self.current_path = path[1:]
                    logger.info(f"Planned path to frontier: {self.current_path}")
                    current_position = self.get_ash_position_grid()
                    next_position = self.current_path[0]
                    action = self.get_action_from_positions(current_position, next_position)
                    logger.critical("known location selected. Returning path action.")
                    logger.critical(f"known = {path}")
                    return action
                else:
                    # Last resort: random action
                    logger.critical("No unexplored maps or transitions left. Returning random action.")
                    if self.current_frontier:
                        self.old_frontier = self.current_frontier
                    self.current_frontier = None
                    self.current_path = []
                    self.steps_since_last_frontier = 0
                    self.steps_towards_current_frontier = 0
                    self.goal_count = 500000
                    self.random_action_steps_remaining = 1
                    return None

    def move_to_unexplored_map(self):
        """
        Plan a path to an unexplored map transition point.
        
        This method is a key component of the exploration strategy, responsible for:
        1. Identifying available transition points from the current map
        2. Selecting a transition point to explore based on distance and exploration history
        3. Setting up path planning to the selected transition
        4. Managing the frontier exploration system
        
        The method implements a cycling strategy for transition selection by:
        - Tracking the last transition index to ensure all transitions are explored
        - Sorting transitions by distance (prioritizing farther transitions)
        - Resetting the index when all transitions have been attempted
        
        Args:
            None
            
        Returns:
            bool: True if a path to an unexplored map was successfully planned,
                  False if no valid transitions were found or path planning failed
            
        Side effects:
        - Updates self.frontiers and self.current_frontier
        - Sets up self.current_path for navigation
        - Resets frontier tracking counters
        
        Dependencies:
        - Calls get_ash_position_grid() to determine current position
        - Calls plan_path() to generate a path to the selected transition
        - Uses self.transitions dictionary to identify map transition points
        - Logs detailed information about the transition selection process
        """
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
        """
        Plan a path to a known area with high curiosity value.
        
        This method is used when the agent needs to navigate back to previously 
        explored areas, prioritizing locations based on:
        1. Curiosity value - areas that might contain interesting features
        2. Distance - preferring areas that are further away (for exploration)
        
        The method is an important part of the exploration strategy that:
        - Balances exploration of new areas with revisiting promising ones
        - Helps the agent escape local minima in exploration
        - Drives continued discovery even after initial exploration
        - Leverages the curiosity mapping to guide intelligent exploration
        
        Args:
            None
            
        Returns:
            List or None: A planned path to the selected location if found,
                         None if no known positions exist or no valid path can be generated
                         
        Potential improvements:
        - Add weighting parameters to balance curiosity vs. distance
        - Consider adding a time component (how recently the area was visited)
        - Implement more sophisticated target selection algorithms
        - Add fallback strategies when no high-curiosity areas exist
        """
        map_id = self.current_map_id
        ash_grid_x, ash_grid_y = self.get_ash_position_grid()[1:]
        known_positions = [(x, y) for (x, y), v in self.map[map_id].items() if v == 1]
        if not known_positions:
            return None  # Return None instead of False
        # Sort known positions by curiosity value (highest to lowest) and distance (furthest to closest)
        known_positions.sort(key=lambda pos: (
            self.curiosity_map[map_id].get(pos, 0),
            -(abs(pos[0] - ash_grid_x) + abs(pos[1] - ash_grid_y))
        ), reverse=True)
        
        # Select the position with the highest curiosity value
        target_position = known_positions[0]
        self.frontiers = {(map_id, target_position[0], target_position[1])}
        self.current_frontier = (map_id, target_position[0], target_position[1])
        self.steps_since_last_frontier = 0
        self.steps_towards_current_frontier = 0
        path = self.plan_path((map_id, ash_grid_x, ash_grid_y), (map_id, target_position[0], target_position[1]))
        if path and len(path) > 1:
            logger.critical(f"Moving towards area with highest curiosity. Path: {path[1:]}")
            return path  # Return the path instead of True
        return None  # Return None if path is None or has length <= 1

    def plan_path(self, start, goal):
        """
        Plan an optimal path from start position to goal position using bidirectional A* search.
        
        This method implements a bidirectional A* pathfinding algorithm optimized for the
        Pokemon game world. It searches simultaneously from both the start and goal positions
        until the searches meet, creating an efficient path that avoids obstacles and prefers
        areas with higher curiosity values.
        
        The bidirectional approach significantly improves performance over traditional A*,
        especially in large open areas. The algorithm also incorporates:
        - Curiosity-based path weighting to prefer unexplored areas
        - Obstacle avoidance based on the SLAM-generated map
        - Map transition point detection and handling
        - Path validation to ensure the path only includes valid moves
        
        Args:
            start (Tuple[int, int, int]): Starting position as (map_id, x, y)
            goal (Tuple[int, int, int]): Goal position as (map_id, x, y)
            
        Returns:
            List[Tuple[int, int, int]] or None: 
                Complete path from start to goal as a list of positions,
                or None if no valid path exists
                
        Dependencies:
        - Uses get_neighbors() to find adjacent valid positions
        - Uses self.map to check for obstacles and free spaces
        - Incorporates curiosity_map for exploration preference
        - Requires is_transition_position() to avoid map transitions in path
        """
        start_map_id, start_x, start_y = start
        goal_map_id, goal_x, goal_y = goal
        
        # Ensure start and goal are on the same map (can't path across maps)
        if start_map_id != goal_map_id:
            return None

        # Return trivial path if start and goal are the same
        if start == goal:
            return [start]

        # Initialize forward search (from start to goal)
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
            """
            Calculate heuristic distance between two points using Manhattan distance.
            
            This function provides an admissible heuristic for the A* search algorithm,
            using the Manhattan distance (sum of x and y differences) multiplied by
            a scaling factor. The scaling factor is set to 0 to make the algorithm
            behave more like Dijkstra's algorithm, which is more thorough but less
            directed than A* with a strong heuristic.
            
            Args:
                a (Tuple[int, int]): First position (x, y)
                b (Tuple[int, int]): Second position (x, y)
                
            Returns:
                float: Heuristic distance estimate between points
            """
            return (abs(a[0] - b[0]) + abs(a[1] - b[1])) * 0

        search_counter = 0
        meeting_node = None

        # Main search loop - continue until forward and backward searches meet
        while open_set_fwd and open_set_bwd:
            # Forward search step
            if open_set_fwd:
                current_f_fwd, current_fwd = heapq.heappop(open_set_fwd)
                search_counter += 1
                
                # Safety check to prevent excessive computation
                if search_counter > max_search_area:
                    logger.warning("Pathfinding aborted: search area too large.")
                    return None
                    
                # Skip if already processed
                if current_fwd in closed_set_fwd:
                    continue
                closed_set_fwd.add(current_fwd)

                # Check if forward search reached backward search frontier
                if current_fwd in closed_set_bwd:
                    meeting_node = current_fwd
                    break

                # Process neighbors in forward direction
                neighbors = self.get_neighbors(current_fwd)
                for nx, ny in neighbors:
                    neighbor = (nx, ny)
                    
                    # Skip if already processed
                    if neighbor in closed_set_fwd:
                        continue
                        
                    # Check if the cell is an obstacle or unvisited
                    cell_value = self.map[start_map_id].get(neighbor, 0)
                    if cell_value in {-1}:  # Skip obstacles
                        continue
                        
                    # Skip map transitions to keep path on same map
                    if self.is_transition_position(start_map_id, nx, ny):
                        continue
                        
                    # Calculate cost with curiosity penalty (lower curiosity = higher cost)
                    curiosity_penalty = 0 - self.curiosity_map[start_map_id].get(neighbor, 1.0)
                    tentative_g = g_score_fwd[current_fwd] + 1 + curiosity_penalty
                    
                    # Update path if found a better one
                    if tentative_g < g_score_fwd[neighbor]:
                        came_from_fwd[neighbor] = current_fwd
                        g_score_fwd[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, (goal_x, goal_y))
                        heapq.heappush(open_set_fwd, (f_score, neighbor))

            # Backward search step (similar to forward)
            if open_set_bwd:
                current_f_bwd, current_bwd = heapq.heappop(open_set_bwd)
                search_counter += 1
                
                # Safety check to prevent excessive computation
                if search_counter > max_search_area:
                    logger.warning("Pathfinding aborted: search area too large.")
                    return None
                    
                # Skip if already processed
                if current_bwd in closed_set_bwd:
                    continue
                closed_set_bwd.add(current_bwd)

                # Check if backward search reached forward search frontier
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
        """
        Determine the action needed to move from current position to next position.
        
        This method is a critical component of the pathfinding system, translating
        the spatial relationship between two positions into a concrete game action.
        It calculates the direction of movement by comparing coordinates and selects
        the appropriate action to move in that direction.
        
        The method handles:
        - Cardinal direction movements (up, down, left, right)
        - Error detection for invalid movements (like diagonals)
        - Fallback to random actions when the movement is invalid
        
        Args:
            current (Tuple[int, int, int]): Current position as (map_id, x, y)
            next_pos (Tuple[int, int, int]): Target position as (map_id, x, y)
            
        Returns:
            int: The action to take:
                - 1: Move right
                - 2: Move left
                - 3: Move up
                - 4: Move down
                - self.ppo_action: Fallback for invalid movements
                
        Dependencies:
        - Used by slam_action() to follow planned paths
        - Falls back to self.ppo_action when movement is invalid
        - Resets frontiers and paths when unexpected movements are detected
        """
        _, cx, cy = current
        _, nx, ny = next_pos
        dx = nx - cx
        dy = ny - cy
        logger.debug(f"Current position: ({cx}, {cy}), Next position: ({nx}, {ny}), dx: {dx}, dy: {dy}")
        
        # Check for invalid diagonal movement
        if dx in [-1, 1] and dy in [-1, 1]:
            self.random_action_steps_remaining = 1
            action = self.ppo_action
            logger.critical(f"Invalid diagonal move from {current} to {next_pos}. Fallback action: {action}")
            return action

        # Determine action based on direction
        if dx == 1 and dy == 0:
            action = 1  # Move Right
        elif dx == -1 and dy == 0:
            action = 2  # Move Left
        elif dx == 0 and dy == -1:
            action = 3  # Move Up
        elif dx == 0 and dy == 1:
            action = 4  # Move Down
        else:
            # Handle unexpected movement (should not happen with valid paths)
            logger.critical(f"Unexpected movement from {current} to {next_pos}. Resetting frontiers.")
            self.frontiers = set()
            if self.current_frontier:
                self.old_frontier = self.current_frontier
            self.current_frontier = None
            self.current_path = []
            self.random_action_steps_remaining = 1
            action = self.ppo_action

        logger.debug(f"Selected action: {action}")
        return action

    def check_collision(self, position: Tuple[int, int, int]) -> bool:
        """
        Check if a given position would result in a collision with an obstacle.
        
        This method determines whether a specific position in the game world
        is traversable by:
        1. Verifying the map ID exists in the offset dictionary
        2. Converting game coordinates to grid coordinates using map offsets
        3. Checking if the position is on a different map than the agent
        4. Examining the cell value in the map data structure
        5. Comparing the position with the agent's current position
        
        Collision detection is a critical component of the navigation system,
        allowing the agent to:
        - Avoid obstacles and impassable terrain
        - Build accurate maps of the environment
        - Make informed pathfinding decisions
        - Understand the boundaries of explorable areas
        
        Args:
            position (Tuple[int, int, int]): A tuple containing (map_id, x, y) coordinates
            
        Returns:
            bool: True if a collision would occur (position is invalid/blocked),
                  False if the position is valid and traversable
                  
        Potential improvements:
        - Cache results for frequently checked positions
        - Add different levels of collision (e.g., slow terrain vs. impassable)
        - Consider adding a time component for dynamic obstacles
        - Handle edge cases like NPCs or movable objects
        """
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
        """
        Calculate the position that would result from taking an action at a given position.
        
        This method predicts where Ash would end up after taking a specific action
        from a given position, without actually executing the action. It's used for:
        - Collision detection by predicting movement before it happens
        - Path planning to determine the next position in a sequence
        - Mapping to identify potential obstacles when movement fails
        
        The method translates directional actions (up, down, left, right) into
        coordinate changes, maintaining the same map ID.
        
        Args:
            position (Tuple[int, int, int]): Starting position as (map_id, x, y)
            action (int): Action to take:
                - 1: Move right (x+1)
                - 2: Move left (x-1)
                - 3: Move up (y-1)
                - 4: Move down (y+1)
                - Others: No movement
                
        Returns:
            Tuple[int, int, int]: The predicted position after taking the action
            
        Dependencies:
        - Used by check_collision() to detect potential collisions
        - Used by update_slam_map() to track attempted movements
        """
        map_id, x, y = position
        dx, dy = {1: (1, 0), 2: (-1, 0), 3: (0, -1), 4: (0, 1)}.get(action, (0, 0))
        attempted_x = x + dx
        attempted_y = y + dy
        return (map_id, attempted_x, attempted_y)

    def get_ash_position_grid(self) -> Tuple[int, int, int]:
        """
        Convert Ash's current game position to grid coordinates used for mapping.
        
        This method translates the raw game coordinates (which are relative to each map)
        into a consistent grid coordinate system used for the SLAM mapping. It:
        1. Gets Ash's current position from the game memory
        2. Ensures the map offset is initialized for the current map
        3. Applies the appropriate offset to convert to grid coordinates
        
        The grid coordinate system is crucial for:
        - Building a consistent map across different game areas
        - Tracking explored and unexplored areas
        - Pathfinding and navigation
        - Detecting frontiers and obstacles
        
        Args:
            None
            
        Returns:
            Tuple[int, int, int]: Ash's position in grid coordinates as (map_id, grid_x, grid_y)
            
        Dependencies:
        - Uses get_ash_position() to get the raw game coordinates
        - Uses map_offsets to convert between coordinate systems
        - Calls initialize_map_offset() if needed for a new map
        """
        map_id, x, y = self.ash_position
        self.initialize_map_offset(map_id, x, y)
        offset_x, offset_y = self.map_offsets[map_id]
        grid_x = x + offset_x
        grid_y = y + offset_y
        logger.debug(f"Ash's grid position: Map ID {map_id}, Grid Position ({grid_x}, {grid_y})")
        return (map_id, grid_x, grid_y)

    def get_pokemon_sprite(self, dexno: int) -> int:
        dexno_str = f"{dexno:03}"
        pokemon_dict = {'001': {'ID': '112', 'Name': 'Rhydon', 'Type1': 'Ground', 'Type2': 'Rock'},
                        '002': {'ID': '115', 'Name': 'Kangaskhan', 'Type1': 'Normal', 'Type2': 'Normal'},
                        '154': {'ID': '3', 'Name': 'Venusaur', 'Type1': 'Grass', 'Type2': 'Poison'},
                        '176': {'ID': '4', 'Name': 'Charmander', 'Type1': 'Fire', 'Type2': 'Fire'},
                        '178': {'ID': '5', 'Name': 'Charmeleon', 'Type1': 'Fire', 'Type2': 'Fire'},
                        '180': {'ID': '6', 'Name': 'Charizard', 'Type1': 'Fire', 'Type2': 'Flying'},
                        '177': {'ID': '7', 'Name': 'Squirtle', 'Type1': 'Water', 'Type2': 'Water'},
                        '179': {'ID': '8', 'Name': 'Wartortle', 'Type1': 'Water', 'Type2': 'Water'},
                        '028': {'ID': '9', 'Name': 'Blastoise', 'Type1': 'Water', 'Type2': 'Water'},
                        '123': {'ID': '10', 'Name': 'Caterpie', 'Type1': 'Bug', 'Type2': 'Bug'},
                        '124': {'ID': '11', 'Name': 'Metapod', 'Type1': 'Bug', 'Type2': 'Bug'},
                        '125': {'ID': '12', 'Name': 'Butterfree', 'Type1': 'Bug', 'Type2': 'Flying'},
                        '112': {'ID': '13', 'Name': 'Weedle', 'Type1': 'Bug', 'Type2': 'Poison'},
                        '113': {'ID': '14', 'Name': 'Kakuna', 'Type1': 'Bug', 'Type2': 'Poison'},
                        '114': {'ID': '15', 'Name': 'Beedrill', 'Type1': 'Bug', 'Type2': 'Poison'},
                        '036': {'ID': '16', 'Name': 'Pidgey', 'Type1': 'Normal', 'Type2': 'Flying'},
                        '150': {'ID': '17', 'Name': 'Pidgeotto', 'Type1': 'Normal', 'Type2': 'Flying'},
                        '151': {'ID': '18', 'Name': 'Pidgeot', 'Type1': 'Normal', 'Type2': 'Flying'},
                        '165': {'ID': '19', 'Name': 'Rattata', 'Type1': 'Normal', 'Type2': 'Normal'},
                        '166': {'ID': '20', 'Name': 'Raticate', 'Type1': 'Normal', 'Type2': 'Normal'},
                        '005': {'ID': '21', 'Name': 'Spearow', 'Type1': 'Normal', 'Type2': 'Flying'},
                        '035': {'ID': '22', 'Name': 'Fearow', 'Type1': 'Normal', 'Type2': 'Flying'},
                        '108': {'ID': '23', 'Name': 'Ekans', 'Type1': 'Poison', 'Type2': 'Poison'},
                        '045': {'ID': '24', 'Name': 'Arbok', 'Type1': 'Poison', 'Type2': 'Poison'},
                        '084': {'ID': '25', 'Name': 'Pikachu', 'Type1': 'Electric', 'Type2': 'Electric'},
                        '085': {'ID': '26', 'Name': 'Raichu', 'Type1': 'Electric', 'Type2': 'Electric'},
                        '096': {'ID': '27', 'Name': 'Sandshrew', 'Type1': 'Ground', 'Type2': 'Ground'},
                        '097': {'ID': '28', 'Name': 'Sandslash', 'Type1': 'Ground', 'Type2': 'Ground'},
                        '015': {'ID': '29', 'Name': 'Nidoran♀', 'Type1': 'Poison', 'Type2': 'Poison'},
                        '168': {'ID': '30', 'Name': 'Nidorina', 'Type1': 'Poison', 'Type2': 'Poison'},
                        '016': {'ID': '31', 'Name': 'Nidoqueen', 'Type1': 'Poison', 'Type2': 'Ground'},
                        '003': {'ID': '32', 'Name': 'Nidoran♂', 'Type1': 'Poison', 'Type2': 'Poison'},
                        '167': {'ID': '33', 'Name': 'Nidorino', 'Type1': 'Poison', 'Type2': 'Poison'},
                        '007': {'ID': '34', 'Name': 'Nidoking', 'Type1': 'Poison', 'Type2': 'Ground'},
                        '004': {'ID': '35', 'Name': 'Clefairy', 'Type1': 'Normal', 'Type2': 'Normal'},
                        '142': {'ID': '36', 'Name': 'Clefable', 'Type1': 'Normal', 'Type2': 'Normal'},
                        '082': {'ID': '37', 'Name': 'Vulpix', 'Type1': 'Fire', 'Type2': 'Fire'},
                        '083': {'ID': '38', 'Name': 'Ninetales', 'Type1': 'Fire', 'Type2': 'Fire'},
                        '100': {'ID': '39', 'Name': 'Jigglypuff', 'Type1': 'Normal', 'Type2': 'Normal'},
                        '101': {'ID': '40', 'Name': 'Wigglytuff', 'Type1': 'Normal', 'Type2': 'Normal'},
                        '107': {'ID': '41', 'Name': 'Zubat', 'Type1': 'Poison', 'Type2': 'Flying'},
                        '130': {'ID': '42', 'Name': 'Golbat', 'Type1': 'Poison', 'Type2': 'Flying'},
                        '185': {'ID': '43', 'Name': 'Oddish', 'Type1': 'Grass', 'Type2': 'Poison'},
                        '186': {'ID': '44', 'Name': 'Gloom', 'Type1': 'Grass', 'Type2': 'Poison'},
                        '187': {'ID': '45', 'Name': 'Vileplume', 'Type1': 'Grass', 'Type2': 'Poison'},
                        '109': {'ID': '46', 'Name': 'Paras', 'Type1': 'Bug', 'Type2': 'Grass'},
                        '046': {'ID': '47', 'Name': 'Parasect', 'Type1': 'Bug', 'Type2': 'Grass'},
                        '065': {'ID': '48', 'Name': 'Venonat', 'Type1': 'Bug', 'Type2': 'Poison'},
                        '119': {'ID': '49', 'Name': 'Venomoth', 'Type1': 'Bug', 'Type2': 'Poison'},
                        '059': {'ID': '50', 'Name': 'Diglett', 'Type1': 'Ground', 'Type2': 'Ground'},
                        '118': {'ID': '51', 'Name': 'Dugtrio', 'Type1': 'Ground', 'Type2': 'Ground'},
                        '077': {'ID': '52', 'Name': 'Meowth', 'Type1': 'Normal', 'Type2': 'Normal'},
                        '144': {'ID': '53', 'Name': 'Persian', 'Type1': 'Normal', 'Type2': 'Normal'},
                        '047': {'ID': '54', 'Name': 'Psyduck', 'Type1': 'Water', 'Type2': 'Water'},
                        '128': {'ID': '55', 'Name': 'Golduck', 'Type1': 'Water', 'Type2': 'Water'},
                        '057': {'ID': '56', 'Name': 'Mankey', 'Type1': 'Fighting', 'Type2': 'Fighting'},
                        '117': {'ID': '57', 'Name': 'Primeape', 'Type1': 'Fighting', 'Type2': 'Fighting'},
                        '033': {'ID': '58', 'Name': 'Growlithe', 'Type1': 'Fire', 'Type2': 'Fire'},
                        '020': {'ID': '59', 'Name': 'Arcanine', 'Type1': 'Fire', 'Type2': 'Fire'},
                        '071': {'ID': '60', 'Name': 'Poliwag', 'Type1': 'Water', 'Type2': 'Water'},
                        '110': {'ID': '61', 'Name': 'Poliwhirl', 'Type1': 'Water', 'Type2': 'Water'},
                        '111': {'ID': '62', 'Name': 'Poliwrath', 'Type1': 'Water', 'Type2': 'Fighting'},
                        '148': {'ID': '63', 'Name': 'Abra', 'Type1': 'Psychic', 'Type2': 'Psychic'},
                        '038': {'ID': '64', 'Name': 'Kadabra', 'Type1': 'Psychic', 'Type2': 'Psychic'},
                        '149': {'ID': '65', 'Name': 'Alakazam', 'Type1': 'Psychic', 'Type2': 'Psychic'},
                        '106': {'ID': '66', 'Name': 'Machop', 'Type1': 'Fighting', 'Type2': 'Fighting'},
                        '041': {'ID': '67', 'Name': 'Machoke', 'Type1': 'Fighting', 'Type2': 'Fighting'},
                        '126': {'ID': '68', 'Name': 'Machamp', 'Type1': 'Fighting', 'Type2': 'Fighting'},
                        '188': {'ID': '69', 'Name': 'Bellsprout', 'Type1': 'Grass', 'Type2': 'Poison'},
                        '189': {'ID': '70', 'Name': 'Weepinbell', 'Type1': 'Grass', 'Type2': 'Poison'},
                        '190': {'ID': '71', 'Name': 'Victreebel', 'Type1': 'Grass', 'Type2': 'Poison'},
                        '024': {'ID': '72', 'Name': 'Tentacool', 'Type1': 'Water', 'Type2': 'Poison'},
                        '155': {'ID': '73', 'Name': 'Tentacruel', 'Type1': 'Water', 'Type2': 'Poison'},
                        '169': {'ID': '74', 'Name': 'Geodude', 'Type1': 'Rock', 'Type2': 'Ground'},
                        '039': {'ID': '75', 'Name': 'Graveler', 'Type1': 'Rock', 'Type2': 'Ground'},
                        '049': {'ID': '76', 'Name': 'Golem', 'Type1': 'Rock', 'Type2': 'Ground'},
                        '163': {'ID': '77', 'Name': 'Ponyta', 'Type1': 'Fire', 'Type2': 'Fire'},
                        '164': {'ID': '78', 'Name': 'Rapidash', 'Type1': 'Fire', 'Type2': 'Fire'},
                        '037': {'ID': '79', 'Name': 'Slowpoke', 'Type1': 'Water', 'Type2': 'Psychic'},
                        '008': {'ID': '80', 'Name': 'Slowbro', 'Type1': 'Water', 'Type2': 'Psychic'},
                        '173': {'ID': '81', 'Name': 'Magnemite', 'Type1': 'Electric', 'Type2': 'Electric'},
                        '054': {'ID': '82', 'Name': 'Magneton', 'Type1': 'Electric', 'Type2': 'Electric'},
                        '064': {'ID': '83', 'Name': "Farfetch'd", 'Type1': 'Normal', 'Type2': 'Flying'},
                        '070': {'ID': '84', 'Name': 'Doduo', 'Type1': 'Normal', 'Type2': 'Flying'},
                        '116': {'ID': '85', 'Name': 'Dodrio', 'Type1': 'Normal', 'Type2': 'Flying'},
                        '058': {'ID': '86', 'Name': 'Seel', 'Type1': 'Water', 'Type2': 'Water'},
                        '120': {'ID': '87', 'Name': 'Dewgong', 'Type1': 'Water', 'Type2': 'Ice'},
                        '013': {'ID': '88', 'Name': 'Grimer', 'Type1': 'Poison', 'Type2': 'Poison'},
                        '136': {'ID': '89', 'Name': 'Muk', 'Type1': 'Poison', 'Type2': 'Poison'},
                        '023': {'ID': '90', 'Name': 'Shellder', 'Type1': 'Water', 'Type2': 'Water'},
                        '139': {'ID': '91', 'Name': 'Cloyster', 'Type1': 'Water', 'Type2': 'Ice'},
                        '025': {'ID': '92', 'Name': 'Gastly', 'Type1': 'Ghost', 'Type2': 'Poison'},
                        '147': {'ID': '93', 'Name': 'Haunter', 'Type1': 'Ghost', 'Type2': 'Poison'},
                        '014': {'ID': '94', 'Name': 'Gengar', 'Type1': 'Ghost', 'Type2': 'Poison'},
                        '034': {'ID': '95', 'Name': 'Onix', 'Type1': 'Rock', 'Type2': 'Ground'},
                        '048': {'ID': '96', 'Name': 'Drowzee', 'Type1': 'Psychic', 'Type2': 'Psychic'},
                        '129': {'ID': '97', 'Name': 'Hypno', 'Type1': 'Psychic', 'Type2': 'Psychic'},
                        '078': {'ID': '98', 'Name': 'Krabby', 'Type1': 'Water', 'Type2': 'Water'},
                        '138': {'ID': '99', 'Name': 'Kingler', 'Type1': 'Water', 'Type2': 'Water'},
                        '006': {'ID': '100', 'Name': 'Voltorb', 'Type1': 'Electric', 'Type2': 'Electric'},
                        '141': {'ID': '101', 'Name': 'Electrode', 'Type1': 'Electric', 'Type2': 'Electric'},
                        '012': {'ID': '102', 'Name': 'Exeggcute', 'Type1': 'Grass', 'Type2': 'Psychic'},
                        '010': {'ID': '103', 'Name': 'Exeggutor', 'Type1': 'Grass', 'Type2': 'Psychic'},
                        '017': {'ID': '104', 'Name': 'Cubone', 'Type1': 'Ground', 'Type2': 'Ground'},
                        '145': {'ID': '105', 'Name': 'Marowak', 'Type1': 'Ground', 'Type2': 'Ground'},
                        '043': {'ID': '106', 'Name': 'Hitmonlee', 'Type1': 'Fighting', 'Type2': 'Fighting'},
                        '044': {'ID': '107', 'Name': 'Hitmonchan', 'Type1': 'Fighting', 'Type2': 'Fighting'},
                        '011': {'ID': '108', 'Name': 'Lickitung', 'Type1': 'Normal', 'Type2': 'Normal'},
                        '055': {'ID': '109', 'Name': 'Koffing', 'Type1': 'Poison', 'Type2': 'Poison'},
                        '143': {'ID': '110', 'Name': 'Weezing', 'Type1': 'Poison', 'Type2': 'Poison'},
                        '018': {'ID': '111', 'Name': 'Rhyhorn', 'Type1': 'Ground', 'Type2': 'Rock'},
                        '040': {'ID': '113', 'Name': 'Chansey', 'Type1': 'Normal', 'Type2': 'Normal'},
                        '030': {'ID': '114', 'Name': 'Tangela', 'Type1': 'Grass', 'Type2': 'Grass'},
                        '092': {'ID': '116', 'Name': 'Horsea', 'Type1': 'Water', 'Type2': 'Water'},
                        '093': {'ID': '117', 'Name': 'Seadra', 'Type1': 'Water', 'Type2': 'Water'},
                        '157': {'ID': '118', 'Name': 'Goldeen', 'Type1': 'Water', 'Type2': 'Water'},
                        '158': {'ID': '119', 'Name': 'Seaking', 'Type1': 'Water', 'Type2': 'Water'},
                        '027': {'ID': '120', 'Name': 'Staryu', 'Type1': 'Water', 'Type2': 'Water'},
                        '152': {'ID': '121', 'Name': 'Starmie', 'Type1': 'Water', 'Type2': 'Psychic'},
                        '042': {'ID': '122', 'Name': 'Mr. Mime', 'Type1': 'Psychic', 'Type2': 'Psychic'},
                        '026': {'ID': '123', 'Name': 'Scyther', 'Type1': 'Bug', 'Type2': 'Flying'},
                        '072': {'ID': '124', 'Name': 'Jynx', 'Type1': 'Ice', 'Type2': 'Psychic'},
                        '053': {'ID': '125', 'Name': 'Electabuzz', 'Type1': 'Electric', 'Type2': 'Electric'},
                        '051': {'ID': '126', 'Name': 'Magmar', 'Type1': 'Fire', 'Type2': 'Fire'},
                        '029': {'ID': '127', 'Name': 'Pinsir', 'Type1': 'Bug', 'Type2': 'Bug'},
                        '060': {'ID': '128', 'Name': 'Tauros', 'Type1': 'Normal', 'Type2': 'Normal'},
                        '133': {'ID': '129', 'Name': 'Magikarp', 'Type1': 'Water', 'Type2': 'Water'},
                        '022': {'ID': '130', 'Name': 'Gyarados', 'Type1': 'Water', 'Type2': 'Flying'},
                        '019': {'ID': '131', 'Name': 'Lapras', 'Type1': 'Water', 'Type2': 'Ice'},
                        '076': {'ID': '132', 'Name': 'Ditto', 'Type1': 'Normal', 'Type2': 'Normal'},
                        '102': {'ID': '133', 'Name': 'Eevee', 'Type1': 'Normal', 'Type2': 'Normal'},
                        '105': {'ID': '134', 'Name': 'Vaporeon', 'Type1': 'Water', 'Type2': 'Water'},
                        '104': {'ID': '135', 'Name': 'Jolteon', 'Type1': 'Electric', 'Type2': 'Electric'},
                        '103': {'ID': '136', 'Name': 'Flareon', 'Type1': 'Fire', 'Type2': 'Fire'},
                        '170': {'ID': '137', 'Name': 'Porygon', 'Type1': 'Normal', 'Type2': 'Normal'},
                        '098': {'ID': '138', 'Name': 'Omanyte', 'Type1': 'Rock', 'Type2': 'Water'},
                        '099': {'ID': '139', 'Name': 'Omastar', 'Type1': 'Rock', 'Type2': 'Water'},
                        '090': {'ID': '140', 'Name': 'Kabuto', 'Type1': 'Rock', 'Type2': 'Water'},
                        '091': {'ID': '141', 'Name': 'Kabutops', 'Type1': 'Rock', 'Type2': 'Water'},
                        '171': {'ID': '142', 'Name': 'Aerodactyl', 'Type1': 'Rock', 'Type2': 'Flying'},
                        '132': {'ID': '143', 'Name': 'Snorlax', 'Type1': 'Normal', 'Type2': 'Normal'},
                        '074': {'ID': '144', 'Name': 'Articuno', 'Type1': 'Ice', 'Type2': 'Flying'},
                        '075': {'ID': '145', 'Name': 'Zapdos', 'Type1': 'Electric', 'Type2': 'Flying'},
                        '073': {'ID': '146', 'Name': 'Moltres', 'Type1': 'Fire', 'Type2': 'Flying'},
                        '088': {'ID': '147', 'Name': 'Dratini', 'Type1': 'Dragon', 'Type2': 'Dragon'},
                        '089': {'ID': '148', 'Name': 'Dragonair', 'Type1': 'Dragon', 'Type2': 'Dragon'},
                        '066': {'ID': '149', 'Name': 'Dragonite', 'Type1': 'Dragon', 'Type2': 'Flying'},
                        '131': {'ID': '150', 'Name': 'Mewtwo', 'Type1': 'Psychic', 'Type2': 'Psychic'},
                        '021': {'ID': '151', 'Name': 'Mew', 'Type1': 'Psychic', 'Type2': 'Psychic'}
                        }
        pokemon_data = pokemon_dict.get(dexno_str)
        
        if pokemon_data:
            # Return the relevant information as a dictionary
            return {
                'ID': pokemon_data['ID'],
                'Name': pokemon_data['Name'],
                'Type1': pokemon_data['Type1'],
                'Type2': pokemon_data['Type2']
            }
        else:
            # Handle the case where the dexno is not found
            raise ValueError(f"Pokemon with dexno {dexno} not found.")

    def increment_collision_count(self, position):
        """
        Increment the collision counter for a specific position.
        
        This method tracks how many times the agent has collided with
        obstacles at a particular map position. This information is used to:
        1. Identify problematic areas in the map
        2. Help the pathfinding algorithm avoid repeatedly trying impassable routes
        3. Detect when the agent might be stuck
        
        The position is a tuple of (map_id, x, y) which uniquely identifies
        a location across different maps.
        
        Args:
            position (Tuple[int, int, int]): A tuple of (map_id, x, y) identifying the position
            
        Returns:
            None - Updates self.collision_counts dictionary internally
            
        Potential improvements:
        - Add bounds checking for valid positions
        - Include option to cap the maximum count
        - Add timestamping to track when collisions occur
        """
        map_id, x, y = position
        self.collision_counts[(map_id, x, y)] += 1

    def get_collision_count(self, position):
        """
        Get the number of collisions that occurred at a specific position.
        
        This method retrieves the collision counter for a particular map position,
        providing information about how many times the agent has unsuccessfully 
        attempted to move through this position.
        
        The collision count is used by navigation and exploration algorithms to:
        - Adjust pathfinding costs
        - Identify and avoid problematic areas
        - Make decisions about exploration strategies
        
        Args:
            position (Tuple[int, int, int]): A tuple of (map_id, x, y) identifying the position
            
        Returns:
            int: The number of collisions at the specified position
            
        Potential improvements:
        - Add error handling for positions not in the dictionary
        - Return 0 instead of raising KeyError for unknown positions
        - Add option to get collision counts for an entire area
        """
        map_id, x, y = position
        return self.collision_counts[(map_id, x, y)]

    def reset_collision_count(self, position):
        """
        Reset the collision counter for a specific position by removing it.
        
        This method completely removes the collision tracking for a particular
        map position, effectively resetting its count. This is useful when:
        1. A previously impassable area becomes passable
        2. The map has changed in some way
        3. We want to force the agent to reconsider a previously problematic path
        
        The method uses dictionary deletion rather than setting to zero,
        which saves memory by not tracking positions with no collisions.
        
        Args:
            position (Tuple[int, int, int]): A tuple of (map_id, x, y) identifying the position
            
        Returns:
            None - Removes the entry from self.collision_counts dictionary
            
        Potential improvements:
        - Add error handling for positions not in the dictionary
        - Add option to reset counts in an entire area
        - Consider tracking when the reset occurred
        """
        map_id, x, y = position
        del self.collision_counts[(map_id, x, y)]
    
    def release_all_keys(self):
        """
        Release all pressed keys in the game emulator.
        
        This method ensures a clean input state by releasing all possible
        directional and button inputs. It is typically called:
        1. Before executing a new action to prevent input overlap
        2. When needing to reset input state between actions
        3. When transitioning between different control modes
        
        The method releases:
        - All four directional arrows (left, right, up, down)
        - A and B buttons
        - Start and Select buttons may also be released depending on implementation
        
        Args:
            None
            
        Returns:
            None - Side effects are within the emulator input state
            
        Potential improvements:
        - Add error handling in case input release fails
        - Consider adding logging to track input state changes
        - Could be optimized to only release previously pressed keys
        - May need adaptation if emulator input mechanism changes
        """
        # Using button_release method from the PyBoy API
        self.pyboy.button_release("left")
        self.pyboy.button_release("right")
        self.pyboy.button_release("up")
        self.pyboy.button_release("down")
        self.pyboy.button_release("a")
        self.pyboy.button_release("b")
        self.pyboy.button_release("start")
        self.pyboy.button_release("select")
    
    def new_pokemon_found(self):
        """
        Check if a new Pokemon species has been encountered in battle.
        
        This method detects when the player encounters a Pokemon species
        that hasn't been seen before by:
        1. Checking if a battle is active
        2. Reading the Pokemon species ID from memory
        3. Comparing it against the list of previously encountered Pokemon
        
        The method is used for:
        - Reward calculation in exploration and collection goals
        - Tracking Pokedex completion progress
        - Determining when to switch to catch-focused behavior
        
        Args:
            None
            
        Returns:
            int: 1 if a new Pokemon species was found, 0 otherwise
            
        Side effects:
        - Updates self.pkm_fnd with the ID of the newly found Pokemon
            
        Dependencies:
        - Calls chk_battling() to verify battle state
        - Reads memory address 0xcfd9 for Pokemon species ID
        - Uses self.pokemon_found_list to track encountered species
        """
        if self.chk_battling():
            v_1 = self.pyboy.memory[0xcfd9]
            if v_1 not in self.pokemon_found_list:
                self.pkm_fnd = v_1
                return 1
        return 0

    def new_pokemon_caught(self):
        """
        Check if a new Pokemon species has been caught.
        
        This method detects when the player catches a Pokemon species
        that hasn't been caught before by:
        1. Checking if a battle is active
        2. Reading the Pokemon species ID from memory
        3. Comparing it against the list of previously caught Pokemon
        
        The method is used for:
        - Reward calculation in collection-focused goals
        - Tracking Pokemon collection progress
        - Determining when to switch between different goal modes
        
        Args:
            None
            
        Returns:
            int: 1 if a new Pokemon species was caught, 0 otherwise
            
        Side effects:
        - Updates self.pkm_cau with the ID of the newly caught Pokemon
            
        Dependencies:
        - Calls chk_battling() to verify battle state
        - Reads memory address 0xcfd9 for Pokemon species ID
        - Uses self.pokemon_caught_list to track caught species
        """
        if self.chk_battling():
            v_1 = self.pyboy.memory[0xcfd9]
            if v_1 not in self.pokemon_caught_list:
                self.pkm_cau = v_1
                return 1
        return 0

    def get_goal(self):
        """
        Determine the current goal for the agent based on game state.
        
        This method analyzes the current game state to select the most appropriate
        goal for the agent. It considers factors such as:
        1. Battle state (whether in battle with trainer or wild Pokemon)
        2. Pokemon strength relative to encountered enemies
        3. Whether new Pokemon have been found or caught
        4. How long the agent has been potentially stuck
        
        The method assigns one of several goal types:
        - 'red': Focus on battling and training Pokemon
        - 'blue': Focus on exploration and discovering new areas
        - 'magenta': Focus on catching new Pokemon
        - 'green': Use random actions to escape stuck situations
        
        Args:
            None
            
        Returns:
            str: The selected goal type ('red', 'blue', 'magenta', or 'green')
            
        Side effects:
        - Updates self.global_goal with the selected goal
        - Updates goal tracking counters
            
        Dependencies:
        - Calls new_pokemon_found() and new_pokemon_caught()
        - Reads various memory addresses for game state
        - Uses self.ash_stuck_counter to detect stuck situations
        """
        total_lvls = self.current_score
        enemy_lvl = self.previous_enemy_lvl if self.previous_enemy_lvl > 0 else 1
        new_pkm_fnd = self.new_pokemon_found()
        new_pkm_cau = self.new_pokemon_caught()
        num_poke_balls = self.pyboy.memory[0xd31e]

        too_strong = total_lvls > (enemy_lvl * 15)
        too_many_steps = self.ash_stuck_counter > 3000
        no_poke_balls = num_poke_balls == 0

        # Check if Ash is battling
        battle_state = self.pyboy.memory[0xd056]
        player_control = self.pyboy.memory[0xc507]
        
        # Initialize goal tracking variables if they don't exist
        if not hasattr(self, 'goal_count'):
            self.goal_count = 0
        if not hasattr(self, 'rand_count'):
            self.rand_count = 0
            
        # Check if new Pokemon found or caught - highest priority
        if (battle_state == 1) and (new_pkm_fnd == 1 or new_pkm_cau == 1):
            goal = 'magenta'  # Catch new Pokemon
        # Check if in battle with trainer - second priority
        elif battle_state == 2:
            goal = 'red'  # Battle the trainer
        # Check if in battle with wild Pokemon - third priority
        elif battle_state == 1:
            if too_strong:
                goal = 'blue'  # Explore instead of battling if too strong
            else:
                goal = 'red'  # Battle wild Pokemon to gain experience
        # Not in battle - exploration or random behavior
        else:
            if player_control == 126 and battle_state == 0:
                goal = 'red'  # Special case for certain player control states
            elif too_many_steps:
                # Agent might be stuck, use random behavior occasionally
                if self.goal_count <= 500000:
                    goal = 'blue'  # Default to exploration
                    self.goal_count += 1
                    logger.info(f"self.goal_count = {self.goal_count}")
                else:
                    if self.rand_count <= 500:
                        goal = 'green'  # Switch to random behavior to escape
                        self.rand_count += 1
                    else:
                        self.goal_count = 0
                        self.rand_count = 0
                        goal = 'green'
            # Normal exploration mode when not too strong
            elif not too_strong:
                if self.goal_count <= 500000:
                    goal = 'blue'  # Explore the world
                    self.goal_count += 1
                    logger.info(f"self.goal_count = {self.goal_count}")
                else:
                    if self.rand_count <= 500:
                        goal = 'red'  # Switch to battling occasionally
                        self.rand_count += 1
                    else:
                        self.goal_count = 0
                        self.rand_count = 0
                        goal = 'red'
            # Exploration mode when too strong
            elif too_strong and battle_state == 0:
                if self.goal_count <= 500000:
                    goal = 'blue'  # Continue exploration
                    self.goal_count += 1
                    logger.info(f"self.goal_count = {self.goal_count}")
                else:
                    if self.rand_count <= 500:
                        goal = 'green'  # Switch to random behavior occasionally
                        self.rand_count += 1
                    else:
                        self.goal_count = 0
                        self.rand_count = 0
                        goal = 'green'
            else:
                goal = 'green'  # Default to random behavior if no other condition met

        self.global_goal = goal
        return goal

    def add_color_block(self, image, goal):
        """
        Add a colored indicator block to an image based on the current goal.
        
        This method adds a visual indicator to the game screen that shows the
        current goal mode of the agent. Different colors represent different goals:
        - 'red': Red block - Focus on battling and training Pokemon
        - 'blue': Blue block - Focus on exploration and discovering new areas
        - 'magenta': Magenta block - Focus on catching new Pokemon
        - 'green': Green block - Random actions to escape stuck situations
        
        The method creates a small colored square in the top-left corner of the image
        to provide a quick visual reference of the agent's current objective.
        
        Args:
            image (np.ndarray): The input image to add the color block to
            goal (str): The current goal ('red', 'blue', 'magenta', or 'green')
            
        Returns:
            np.ndarray: The modified image with the color block added
            
        Dependencies:
        - Uses OpenCV (cv2) for image manipulation
        - Called by render() and get_observation() methods
        """
        # Create a copy of the image to avoid modifying the original
        result = image.copy()
        
        # Define color based on goal
        if goal == 'red':
            color = (255, 0, 0)  # Red
        elif goal == 'blue':
            color = (0, 0, 255)  # Blue
        elif goal == 'magenta':
            color = (255, 0, 255)  # Magenta
        elif goal == 'green':
            color = (0, 255, 0)  # Green
        else:
            color = (128, 128, 128)  # Gray for unknown goal
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Define block size (10% of the smaller dimension)
        block_size = min(height, width) // 10
        
        # Draw the colored block in the top-left corner
        cv2.rectangle(result, (0, 0), (block_size, block_size), color, -1)
        
        return result
    
    def get_map_id(self):
        """
        Get the current map ID from game memory.
        
        This method reads the map ID value from a specific memory address in the
        Pokemon Yellow game. The map ID is a crucial piece of information used for:
        1. Tracking the player's location in the game world
        2. Organizing the SLAM mapping system by map areas
        3. Detecting transitions between different game areas
        4. Tracking exploration progress
        
        The map ID changes whenever the player enters a new area (building, cave,
        new route, etc.) and is used as a primary key in many of the mapping data
        structures.
        
        Args:
            None
            
        Returns:
            int: The current map ID value from memory address 0xd35d
            
        Dependencies:
        - Reads directly from PyBoy emulator memory
        - Used by navigation, mapping, and exploration systems
        - Critical for maintaining separate maps for different game areas
        """
        return self.pyboy.memory[0xd35d]
    
    def add_to_map(self, position, value):
        """
        Add or update a position in the SLAM map with the specified value.
        
        This method is a key component of the SLAM (Simultaneous Localization And Mapping)
        system. It takes a position in the game world and marks it in the internal
        grid-based map representation with the specified value. The method:
        1. Ensures the map offset is initialized for the current map
        2. Converts game coordinates to grid coordinates using the map offset
        3. Updates the map and curiosity values accordingly
        
        Args:
            position (Tuple[int, int, int]): Position as (map_id, x, y) in game coordinates
            value (int): Value to set at this position:
                         -1 = obstacle/wall
                          1 = free/walkable space
                          0 = unknown (default)
                          
        Returns:
            None - Updates the internal map representation
            
        Side effects:
            - Updates self.map dictionary with new cell values
            - Updates self.curiosity_map with adjusted curiosity values
            - May initialize map offsets if not already set
            
        Dependencies:
            - Requires initialize_map_offset() to ensure offsets are set
            - Uses map_offsets to convert between coordinate systems
        """
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

    def detect_frontiers(self):
        """
        Detect and update the set of frontier cells in the SLAM map.
        
        This method identifies frontier cells - free cells adjacent to unexplored areas -
        which are potential targets for exploration. It:
        1. Gets the agent's current grid position
        2. Checks for target frontier reached
        3. Identifies all possible frontiers based on map data
        4. Filters frontiers based on various criteria (transitions, etc.)
        
        Frontiers are crucial for the exploration strategy as they represent
        the boundary between known and unknown areas of the game world.
        
        Args:
            None
            
        Returns:
            None - Updates the internal frontiers representation
            
        Side effects:
            - Updates self.frontiers with new frontier cells
            - May update target frontier information
            - Logs frontier detection information
            
        Dependencies:
            - Uses get_ash_position_grid() to determine current position
            - Uses check_target_frontier_reached() to update frontier status
            - Uses get_neighbors() to find adjacent cells
            - Uses is_transition_position() to filter out transition points
        """
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
                            break  # Found a frontier adjacent to this cell
    
    def get_ash_position(self):
        """
        Get the current position of Ash (the player character) from game memory.
        
        This method reads the game's memory to determine Ash's current position
        in the game world. It retrieves:
        1. The map ID (which area of the game world Ash is in)
        2. The x-coordinate on the current map
        3. The y-coordinate on the current map
        
        This position data is fundamental for:
        - Tracking the agent's movement through the game
        - Building and updating the SLAM map
        - Determining exploration progress
        - Detecting map transitions
        
        Args:
            None
            
        Returns:
            Tuple[int, int, int]: Ash's position as (map_id, x_coord, y_coord)
            
        Side effects:
            - Updates self.current_map_id with the current map identifier
            
        Dependencies:
            - Requires access to PyBoy's memory interface
            - Reads specific memory addresses where the game stores position data
        """
        y_coord = self.pyboy.memory[0xd360]
        x_coord = self.pyboy.memory[0xd361]
        map_id = self.pyboy.memory[0xd35d]
        self.current_map_id = map_id  # Update current_map_id
        return (map_id, x_coord, y_coord)
    
    def update_slam_map(self):
        """
        Update the SLAM (Simultaneous Localization and Mapping) map based on current position.
        
        This method is called after each movement to update the internal map representation.
        It performs several key functions:
        1. Marks the current position as free space (value 1)
        2. Checks for collisions with obstacles
        3. Detects map transitions
        4. Updates the curiosity map for exploration
        5. Handles stuck detection and recovery
        
        The method is critical for building an accurate representation of the game world
        that can be used for pathfinding and exploration.
        
        Args:
            None
            
        Returns:
            None - Updates internal map data structures
            
        Side effects:
            - Updates self.map with new cell values
            - May mark obstacles when collisions are detected
            - Updates self.ash_stuck_counter for stuck detection
            - May trigger transition handling for map changes
        """
        if self.ash_position is None or self.ash_old_position is None:
            return
            
        # Mark current position as free space
        self.add_to_map(self.ash_position, 1)
        
        # Check if we're at the same position as before (potentially stuck)
        if self.ash_position == self.ash_old_position and self.last_action is not None:
            # We tried to move but didn't change position - might be an obstacle
            attempted_position = self.get_attempted_position(self.ash_old_position, self.last_action)
            
            # Mark the attempted position as an obstacle if we couldn't move there
            if attempted_position != self.ash_position:
                self.add_to_map(attempted_position, -1)
                self.increment_collision_count(attempted_position)
                
            # Increment stuck counter when not moving
            self.ash_stuck_counter += 1
            logger.debug(f"Ash may be stuck. Counter: {self.ash_stuck_counter}")
        else:
            # We moved successfully, reset stuck counter
            self.ash_stuck_counter = 0
            
        # Check for map transitions
        current_map_id = self.ash_position[0]
        if self.ash_old_position[0] != current_map_id:
            # Map transition occurred
            self.transition_positions[self.ash_old_position[0]].add((self.ash_old_position[1], self.ash_old_position[2]))
            logger.info(f"Map transition detected from {self.ash_old_position[0]} to {current_map_id}")
            
        # Update curiosity map - reduce curiosity for visited cells
        map_id, x, y = self.ash_position
        offset_x, offset_y = self.map_offsets.get(map_id, (self.map_offset, self.map_offset))
        grid_x = x + offset_x
        grid_y = y + offset_y
        self.curiosity_map[map_id][(grid_x, grid_y)] = max(
            0.0, self.curiosity_map[map_id][(grid_x, grid_y)] - self.curiosity_decay_rate
        )
    
    def is_ash_stuck(self):
        """
        Check if Ash appears to be stuck in the same position.
        
        This method detects when Ash is potentially stuck by:
        1. Checking if a battle is active (not considered stuck during battles)
        2. Analyzing the history of recent positions
        3. Determining if Ash has been moving between only 1-2 positions
        
        Stuck detection is important for:
        - Providing negative rewards to discourage getting stuck
        - Triggering exploration behaviors to escape stuck situations
        - Switching between different goal modes when progress stalls
        
        Args:
            None
            
        Returns:
            bool: True if Ash appears to be stuck, False otherwise
            
        Side effects:
            - May update self.ash_stuck_counter
            
        Dependencies:
            - Calls chk_battling() to verify battle state
            - Uses self.ash_loc_history to track position history
        """
        # Not considered stuck during battles
        if self.chk_battling():
            return False
            
        stuck = False
        loc = self.get_ash_position()
        self.ash_loc_history.append(loc)
        
        # Check if history is full and contains limited unique positions
        if len(self.ash_loc_history) == self.ash_loc_history.maxlen:
            if len(set(self.ash_loc_history)) <= 2:
                stuck = True
            else:
                # Reset counter if moving between more than 2 positions
                self.ash_stuck_counter = 0
                
        return stuck
    
    def timed_out(self):
        """
        Check if the current episode has exceeded the maximum allowed steps.
        
        This method determines if the episode should be truncated due to
        reaching the maximum episode length. This prevents episodes from
        running indefinitely and ensures training progress.
        
        Args:
            None
            
        Returns:
            bool: True if the episode has timed out, False otherwise
            
        Dependencies:
            - Uses self.episode_length to track current steps
            - Uses self.max_episodes for the timeout threshold
        """
        return self.episode_length >= self.max_episodes
    
    def level_did_progress(self):
        """
        Check if the agent has discovered a new map area.
        
        This method tracks progress through the game by detecting when
        the agent enters a new map area that hasn't been visited before.
        It's used to provide rewards for exploration and discovery.
        
        Args:
            None
            
        Returns:
            int: 1 if a new map area was discovered, 0 otherwise
            
        Side effects:
            - Updates self.level_progress list with newly discovered map IDs
            
        Dependencies:
            - Calls get_map_id() to determine the current map
            - Uses self.level_progress to track visited maps
        """
        current_map_id = self.get_map_id()
        if current_map_id in self.level_progress:
            return 0
        else:
            self.level_progress.append(current_map_id)
            return 1
    