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
import base64
import io

# VLM Integration imports
try:
    import requests
    import yaml
    from vlm_config_loader import VLMConfigLoader, VLMConfig
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    print("Warning: Required packages (requests, yaml) not available. VLM features will be disabled.")

# LLM Controller imports
try:
    import asyncio
    from llm_controller import LLMController
    LLM_AVAILABLE = True
    print("[LLM Import] LLM controller module imported successfully")
except ImportError as e:
    LLM_AVAILABLE = False
    print(f"[LLM Import] ERROR: LLM controller not available. Import error: {e}")
    print("Warning: LLM features will be disabled.")
except Exception as e:
    LLM_AVAILABLE = False
    print(f"[LLM Import] ERROR: Unexpected error importing LLM controller: {e}")
    print("Warning: LLM features will be disabled.")

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
        self.frame_stack = deque(maxlen=3)
        self.frame_skip = 1  # Number of frames to skip
        self.observation_space = Box(low=0, high=255, shape=(120, 120, 9), dtype=np.uint8)
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
        # get all party pokemon
        self.pkm_one = None
        self.pkm_two = None
        self.pkm_three = None
        self.pkm_four = None
        self.pkm_five = None
        self.pkm_six = None
        self.current_box = None
        
        # VLM Integration attributes
        self.vlm_config_loader = VLMConfigLoader() if VLM_AVAILABLE else None
        self.vlm_config = None
        self.vlm_enabled = False
        self.vlm_step_counter = 0
        self.last_vlm_decision = None
        self.vlm_context_history = deque(maxlen=5)  # Store recent context for VLM
        self.vlm_action_override = None  # Override action from VLM
        self.vlm_active_steps = 0  # How many steps VLM control is active
        
        # Load VLM configuration
        self.load_vlm_config()
        
        # LLM Controller attributes - used when no frontiers are detected
        self.llm_controller = None
        self.llm_enabled = False
        self.llm_override_enabled = False  # Toggle for LLM controller override
        self.llm_toggle_button_bounds = None  # Initialize button bounds for mouse click detection
        self.llm_action_history = deque(maxlen=20)  # Store last 20 LLM actions for context
        if LLM_AVAILABLE:
            try:
                print("[LLM Init] Attempting to initialize LLM controller...")
                self.llm_controller = LLMController()
                self.llm_enabled = True
                print("[LLM Init] LLM controller initialized successfully!")
                logger.info("LLM controller initialized successfully")
            except Exception as e:
                error_msg = f"Failed to initialize LLM controller: {e}"
                print(f"[LLM Init] ERROR: {error_msg}")
                print(f"[LLM Init] Exception type: {type(e).__name__}")
                import traceback
                print(f"[LLM Init] Traceback:\n{traceback.format_exc()}")
                logger.warning(error_msg)
                self.llm_enabled = False
        else:
            print("[LLM Init] LLM_AVAILABLE is False - LLM controller module not imported")
        
        # Mouse callback for interactive controls in the map window
        # Create the window and set up mouse callback
        cv2.namedWindow("SLAM Map")
        cv2.setMouseCallback("SLAM Map", self.on_mouse_click)

    def reset_game_state(self):
        self.total_reward = 0
        self.episode_length = 0
        self.current_score = 0
        # Reset LLM action history on episode reset
        if hasattr(self, 'llm_action_history'):
            self.llm_action_history.clear()
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
        self.pkm_one = None
        self.pkm_two = None
        self.pkm_three = None
        self.pkm_four = None
        self.pkm_five = None
        self.pkm_six = None
        self.current_box = None

        # Reset VLM attributes (only if they exist)
        if hasattr(self, 'vlm_step_counter'):
            self.vlm_step_counter = 0
        if hasattr(self, 'last_vlm_decision'):
            self.last_vlm_decision = None
        if hasattr(self, 'vlm_context_history'):
            self.vlm_context_history.clear()
        if hasattr(self, 'vlm_action_override'):
            self.vlm_action_override = None
        if hasattr(self, 'vlm_active_steps'):
            self.vlm_active_steps = 0

    def encode_screen_for_vlm(self) -> str:
        """
        Encode the current game screen as a base64 image for VLM processing.
        
        Returns:
            str: Base64 encoded image string
        """
        if not self.vlm_enabled:
            return None
            
        # Get current screen
        raw_screen = self.pyboy.screen.ndarray
        raw_screen = cv2.cvtColor(raw_screen, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(raw_screen)
        
        # Encode to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_str

    def get_game_context(self) -> dict:
        """
        Extract relevant game context for VLM decision making.
        
        Returns:
            dict: Game state context
        """
        context = {
            'current_goal': self.global_goal,
            'is_battling': self.is_battling_fl,
            'ash_position': self.ash_position,
            'current_map_id': self.current_map_id,
            'ash_hp': self.ash_hp,
            'pokemon_party': [self.pkm_one, self.pkm_two, self.pkm_three, 
                            self.pkm_four, self.pkm_five, self.pkm_six],
            'steps_at_same_position': self.steps_at_same_position,
            'episode_length': self.episode_length,
            'last_action': self.last_action,
            'has_path': bool(self.current_path),
            'current_frontier': self.current_frontier
        }
        return context

    def should_use_vlm(self) -> bool:
        """
        Determine if VLM should be consulted for decision making.
        
        Returns:
            bool: True if VLM should be used
        """
        if not self.vlm_enabled or not self.vlm_config:
            return False
            
        # Use configuration-based conditions
        conditions = []
        
        if self.vlm_config.conditions.use_in_battle and self.is_battling_fl:
            conditions.append(True)
        
        if (self.vlm_config.conditions.use_when_stuck and 
            self.steps_at_same_position > self.vlm_config.conditions.stuck_threshold):
            conditions.append(True)
        
        if self.global_goal in self.vlm_config.conditions.use_for_goals:
            conditions.append(True)
        
        if (self.vlm_config.conditions.use_when_really_stuck and 
            self.ash_stuck_counter > self.vlm_config.conditions.really_stuck_threshold):
            conditions.append(True)
        
        if (self.vlm_config.conditions.periodic_checks and 
            self.vlm_step_counter % self.vlm_config.decision.call_frequency == 0):
            conditions.append(True)
        
        return any(conditions)

    def get_vlm_decision(self, screen_b64: str, context: dict) -> dict:
        """
        Get action decision from VLM based on current screen and context.
        
        Args:
            screen_b64 (str): Base64 encoded screen image
            context (dict): Game state context
            
        Returns:
            dict: VLM decision with action, confidence, and reasoning
        """
        if not self.vlm_enabled or not self.vlm_config:
            return None
            
        try:
            # Build prompt based on context
            prompt = self.build_vlm_prompt(context)
            
            # Prepare the request payload for OpenAI-compatible API
            payload = {
                "model": self.vlm_config.api.model,
                "messages": [
                    {
                        "role": "system",
                        "content": self.vlm_config.prompts.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{screen_b64}"
                                }
                            }
                        ]
                    }
                ],
                **self.vlm_config.api.parameters
            }
            
            # Make request to OpenAI-compatible API
            response = requests.post(
                self.vlm_config_loader.get_api_endpoint(),
                headers=self.vlm_config_loader.get_api_headers(),
                json=payload,
                timeout=self.vlm_config.api.timeout
            )
            
            if response.status_code == 200:
                response_data = response.json()
                vlm_text = response_data['choices'][0]['message']['content']
                
                # Log API call if enabled
                if self.vlm_config.logging.log_api_calls:
                    logger.debug(f"VLM API Response: {vlm_text}")
                
                return self.parse_vlm_response(vlm_text)
            else:
                logger.error(f"VLM API error: {response.status_code} - {response.text}")
                return None
            
        except requests.exceptions.Timeout:
            logger.error("VLM API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"VLM API request error: {e}")
            return None
        except Exception as e:
            logger.error(f"VLM decision error: {e}")
            return None

    def build_vlm_prompt(self, context: dict) -> str:
        """
        Build a contextual prompt for the VLM based on game state.
        
        Args:
            context (dict): Game state context
            
        Returns:
            str: Formatted prompt for VLM
        """
        if not self.vlm_config:
            # Fallback prompt if config not available
            return "Analyze this Pokemon Yellow game screen and suggest the best action (0-5). Respond in JSON format."
        
        # Use configurable context template
        context_str = self.vlm_config.prompts.context_template.format(**context)
        
        # Combine all prompt parts
        full_prompt = f"{context_str}\n\n{self.vlm_config.prompts.action_guidance}"
        
        return full_prompt

    def parse_vlm_response(self, response_text: str) -> dict:
        """
        Parse VLM response text into structured decision.
        
        Args:
            response_text (str): Raw VLM response
            
        Returns:
            dict: Parsed decision with action, confidence, reasoning
        """
        try:
            # Try to extract JSON from response
            import json
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                decision = json.loads(json_str)
                
                # Validate decision structure
                if 'action' in decision and 'confidence' in decision:
                    # Ensure action is within valid range
                    action = max(0, min(5, int(decision['action'])))
                    confidence = max(0.0, min(1.0, float(decision['confidence'])))
                    
                    return {
                        'action': action,
                        'confidence': confidence,
                        'reasoning': decision.get('reasoning', 'No reasoning provided')
                    }
            
            # Fallback parsing if JSON fails
            lines = response_text.lower().split('\n')
            for line in lines:
                if 'action' in line and any(str(i) in line for i in range(6)):
                    action = next(int(i) for i in range(6) if str(i) in line)
                    return {
                        'action': action,
                        'confidence': 0.5,
                        'reasoning': 'Fallback parsing'
                    }
                    
        except Exception as e:
            logger.error(f"Error parsing VLM response: {e}")
            
        return None

    def vlm_action_decision(self) -> Optional[int]:
        """
        Main VLM decision pipeline - determines if and how to use VLM.
        
        Returns:
            Optional[int]: VLM-recommended action or None to use existing logic
        """
        if not self.should_use_vlm():
            return None
            
        # Increment step counter
        self.vlm_step_counter += 1
        
        # If VLM override is still active, continue using it
        if self.vlm_action_override and self.vlm_active_steps > 0:
            self.vlm_active_steps -= 1
            action = self.vlm_action_override
            logger.info(f"Using VLM override action: {action} (steps remaining: {self.vlm_active_steps})")
            return action
        
        # Get current screen and context
        screen_b64 = self.encode_screen_for_vlm()
        if not screen_b64:
            return None
            
        context = self.get_game_context()
        
        # Store context in history
        self.vlm_context_history.append({
            'step': self.episode_length,
            'context': context.copy()
        })
        
        # Get VLM decision
        vlm_decision = self.get_vlm_decision(screen_b64, context)
        
        if vlm_decision and vlm_decision['confidence'] >= self.vlm_config.decision.confidence_threshold:
            action = vlm_decision['action']
            self.last_vlm_decision = vlm_decision
            
            # For high-confidence decisions, use VLM action for multiple steps
            if vlm_decision['confidence'] >= self.vlm_config.decision.high_confidence_threshold:
                self.vlm_action_override = action
                self.vlm_active_steps = self.vlm_config.decision.override_steps
            
            # Log decision if enabled
            if self.vlm_config.logging.log_decisions:
                logger.info(f"VLM decision: action={action}, confidence={vlm_decision['confidence']:.2f}, reasoning={vlm_decision['reasoning']}")
            
            return action
        
        return None

    def step(self, action):
        reward = 0
        done = False
        info = {}
        for _ in range(self.frame_skip):
            if not done:
                self.get_goal()
                self.battling()
                self.random_action_steps_remaining = 0
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
                
                # Check if VLM should override action decision
                vlm_action = self.vlm_action_decision()
                if vlm_action is not None:
                    # Use VLM-recommended action
                    self.last_action = vlm_action
                    self.execute_action(self.last_action)
                    self.pyboy.tick(1, False)
                    if not self.is_battling_fl:
                        self.ash_old_position = self.ash_position
                        self.ash_position = self.get_ash_position()
                        self.ash_loc_history.append(self.ash_position)
                        self.update_slam_map()
                # Check if LLM override is enabled - this takes priority over goal-based logic
                elif self.llm_override_enabled and not self.is_battling_fl:
                    print(f"[LLM Override] Enabled - calling LLM controller...")
                    llm_action = self.get_llm_action()
                    if llm_action is not None:
                        print(f"[LLM Override] LLM returned action: {llm_action}")
                        logger.info(f"LLM override enabled - using LLM action: {llm_action}")
                        self.last_action = llm_action
                        # For movement actions (1-4), execute twice to ensure movement instead of just turning
                        if llm_action in [1, 2, 3, 4]:
                            self.execute_action(self.last_action)
                            self.pyboy.tick(1, False)
                            # Execute the movement action a second time
                            self.execute_action(self.last_action)
                        else:
                            # For button actions (0, 5), execute once
                            self.execute_action(self.last_action)
                        self.pyboy.tick(1, False)
                        # Store this action in history after executing (so next LLM call sees what was done)
                        self.llm_action_history.append(llm_action)
                        if not self.is_battling_fl:
                            self.ash_old_position = self.ash_position
                            self.ash_position = self.get_ash_position()
                            self.ash_loc_history.append(self.ash_position)
                            self.update_slam_map()
                    else:
                        # LLM failed, fallback to PPO action
                        print(f"[LLM Override] LLM failed, using PPO fallback action: {action}")
                        logger.warning("LLM override enabled but LLM action failed. Using PPO fallback.")
                        self.last_action = action
                        self.execute_action(self.last_action)
                        self.pyboy.tick(1, False)
                        if not self.is_battling_fl:
                            self.ash_old_position = self.ash_position
                            self.ash_position = self.get_ash_position()
                            self.ash_loc_history.append(self.ash_position)
                            self.update_slam_map()
                elif self.global_goal == 'blue' and not self.is_battling_fl:
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
        self.pkm_one = self.pyboy.memory[0xD16A]
        self.pkm_two = self.pyboy.memory[0xD196]
        self.pkm_three = self.pyboy.memory[0xD1C2]
        self.pkm_four = self.pyboy.memory[0xD1EE]
        self.pkm_five = self.pyboy.memory[0xD21A]
        self.pkm_six = self.pyboy.memory[0xD246]
        self.current_box = self.pyboy.memory[0xDA7F]
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

        # Draw LLM Controller Override Toggle Button
        self.draw_llm_toggle_button()

        # Display the map with the path
        cv2.imshow("SLAM Map", self.map_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def draw_llm_toggle_button(self):
        """
        Draw an interactive toggle button for LLM controller override in the map window.
        The button is drawn in the top-left corner of the map.
        """
        # Define button position and size
        button_x = 10
        button_y = 10
        button_width = 200
        button_height = 40
        
        # Store button bounds for mouse click detection
        self.llm_toggle_button_bounds = (button_x, button_y, button_x + button_width, button_y + button_height)
        
        # Choose color based on toggle state
        if self.llm_override_enabled:
            button_color = (0, 255, 0)  # Green when enabled
            status_text = "LLM Override: ON"
            text_color = (0, 0, 0)  # Black text
        else:
            button_color = (128, 128, 128)  # Gray when disabled
            status_text = "LLM Override: OFF"
            text_color = (255, 255, 255)  # White text
        
        # Draw button background
        cv2.rectangle(self.map_image, 
                     (button_x, button_y), 
                     (button_x + button_width, button_y + button_height), 
                     button_color, -1)
        
        # Draw button border
        cv2.rectangle(self.map_image, 
                     (button_x, button_y), 
                     (button_x + button_width, button_y + button_height), 
                     (0, 0, 0), 2)
        
        # Draw status text
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = button_x + (button_width - text_size[0]) // 2
        text_y = button_y + (button_height + text_size[1]) // 2
        cv2.putText(self.map_image, status_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Draw instruction text below button
        instruction_text = "Click to toggle"
        instruction_y = button_y + button_height + 15
        cv2.putText(self.map_image, instruction_text, (button_x, instruction_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    def on_mouse_click(self, event, x, y, flags, param):
        """
        Handle mouse click events in the SLAM Map window.
        Toggles LLM override when the toggle button is clicked.
        
        Args:
            event: Mouse event type (cv2.EVENT_LBUTTONDOWN, etc.)
            x, y: Mouse coordinates
            flags: Additional flags
            param: Additional parameters
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is within toggle button bounds
            if hasattr(self, 'llm_toggle_button_bounds'):
                btn_x1, btn_y1, btn_x2, btn_y2 = self.llm_toggle_button_bounds
                if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
                    # Toggle LLM override
                    self.llm_override_enabled = not self.llm_override_enabled
                    status = "enabled" if self.llm_override_enabled else "disabled"
                    logger.info(f"LLM override {status} via mouse click")
                    print(f"[LLM Toggle] LLM Override {status.upper()} - LLM enabled: {self.llm_enabled}, Controller available: {self.llm_controller is not None}")

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
                    if self.rand_count <= 500:
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
                    if self.rand_count <= 500:
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
                    if self.rand_count <= 500:
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

        # Check for and remove isolated map positions
        map_id = current_position[0]
        positions_to_remove = []
        
        for (x, y), value in self.map[map_id].items():
            if value == 1:
                neighbors = [
                    (x-1, y), (x+1, y),
                    (x, y-1), (x, y+1)
                ]
                if all(self.map[map_id].get(pos, 0) == 0 for pos in neighbors):
                    positions_to_remove.append((x, y))
        
        for pos in positions_to_remove:
            del self.map[map_id][pos]
            if pos in self.curiosity_map[map_id]:
                del self.curiosity_map[map_id][pos]
            logger.debug(f"Removed isolated position {pos} from map {map_id}")

        if self.ash_old_position and self.ash_position:

            previous_map_id = self.ash_old_position[0]
            current_map_id = self.ash_position[0]
            if previous_map_id != current_map_id and self.ash_hp != 0:
                # Map transition detected - find the transition position on the previous map
                transition_found = False
                if self.current_frontier and self.current_frontier[0] == previous_map_id:
                    # Use the current frontier if it's on the previous map
                    frontier_map_id, prev_x, prev_y = self.current_frontier
                    transition_found = True
                    logging.critical(f"update_slam_map self.current_frontier = {self.current_frontier}")
                elif self.old_frontier and self.old_frontier[0] == previous_map_id:
                    # Use the old frontier if it's on the previous map
                    frontier_map_id, prev_x, prev_y = self.old_frontier
                    transition_found = True
                    logging.critical(f"update_slam_map self.old_frontier = {self.old_frontier}")
                else:
                    # Try to find the transition position from the old position
                    # Use the old position as the transition point
                    _, prev_x_game, prev_y_game = self.ash_old_position
                    if previous_map_id in self.map_offsets:
                        prev_offset_x, prev_offset_y = self.map_offsets[previous_map_id]
                        prev_x = prev_x_game + prev_offset_x
                        prev_y = prev_y_game + prev_offset_y
                        transition_found = True
                        logging.critical(f"Using old position as transition: ({previous_map_id}, {prev_x}, {prev_y})")
                
                if transition_found:
                    prev_offset_x, prev_offset_y = self.map_offsets[previous_map_id]
                    self.transitions[previous_map_id].add((prev_x, prev_y))
                    self.handle_map_transition(previous_map_id, prev_x, prev_y)
                
                # Clear current frontier and path after map transition
                if self.current_frontier:
                    self.old_frontier = self.current_frontier
                self.current_frontier = None
                self.current_path = []
                # Reset transition index to allow exploring new transitions on the new map
                self.last_transition_index = -1

        if previous_position == current_position and self.frontiers:
            if self.last_action in [1, 2, 3, 4]:
                attempted_position = self.get_attempted_position(previous_position, self.last_action)
                collision = self.check_collision(attempted_position)
                if collision and self.global_goal == "blue" and self.random_action_steps_remaining == 0:
                    self.increment_collision_count(attempted_position)
                    collision_count = self.get_collision_count(attempted_position)
                    logger.debug(f"Collision detected at {attempted_position}. Collision count: {collision_count}")
                    if collision_count >= 15:
                        self.add_to_map(attempted_position, -1)
                        logger.info(f"Marked position {attempted_position} as an obstacle due to repeated collisions.")
                        self.current_path = []
                        self.reset_collision_count(attempted_position)
        else:
            self.add_to_map(previous_position, 1)
            logger.debug(f"Updated map with free position: {previous_position}")

        # self.detect_frontiers()
        return

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
        # Check if LLM override is enabled - if so, bypass all frontier detection
        if self.llm_override_enabled:
            llm_action = self.get_llm_action()
            if llm_action is not None:
                logger.info(f"LLM override enabled - using LLM action: {llm_action}")
                return llm_action
            else:
                # LLM failed, fallback to PPO action
                logger.warning("LLM override enabled but LLM action failed. Using PPO fallback.")
                return self.ppo_action
        
        # Normal SLAM-based action selection (frontier detection, path planning, etc.)
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
                    self.random_action_steps_remaining = 1
                    return self.ppo_action

            # Check if current position is on a different map than expected
            current_map_id = current_position[0]
            next_map_id = next_position[0]
            
            # If we're on a different map, clear the path and let the system handle it
            if current_map_id != next_map_id:
                logger.info(f"Map mismatch detected: current map {current_map_id} vs expected {next_map_id}. Clearing path.")
                self.current_path = []
                self.detect_frontiers()
                if self.current_frontier:
                    # Try to plan a new path on the current map
                    path = self.plan_path(current_position, self.current_frontier)
                    if path and len(path) > 1:
                        self.current_path = path[1:]
                        next_position = self.current_path[0]
                    else:
                        # No path available, return fallback
                        self.random_action_steps_remaining = 1
                        return self.ppo_action
                else:
                    # No frontier available, return fallback
                    self.random_action_steps_remaining = 1
                    return self.ppo_action

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
                    self.random_action_steps_remaining = 1
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
                    logger.critical("No unexplored maps or transitions left. Returning fallback action.")
                    if self.current_frontier:
                        self.old_frontier = self.current_frontier
                    self.current_frontier = None
                    self.current_path = []
                    self.steps_since_last_frontier = 0
                    self.steps_towards_current_frontier = 0
                    self.goal_count = 500000
                    self.random_action_steps_remaining = 1
                    return self.ppo_action

    def move_to_unexplored_map(self):
        current_map_id = self.current_map_id
        transitions = self.transitions.get(current_map_id, set())
        unexplored_transitions = []

        ash_map_id, ash_x_grid, ash_y_grid = self.get_ash_position_grid()
        if not self.current_path:
            # Filter out transitions that are already visited or excluded
            for position in transitions:
                # Check if this transition position is already visited or excluded
                if (current_map_id, position[0], position[1]) in self.visited_frontiers:
                    continue
                if (current_map_id, position[0], position[1]) in self.excluded_frontiers:
                    continue
                logging.critical(f"move to unexplored map triggered")
                cell_x, cell_y = position
                distance = abs(cell_x - ash_x_grid) + abs(cell_y - ash_y_grid)
                unexplored_transitions.append((position, current_map_id, distance))

            if unexplored_transitions:
                logging.critical(f"There are unexplored_transitions")
                # Sort by distance (closest first) instead of reverse
                unexplored_transitions.sort(key=lambda t: t[2])

                num_transitions = len(unexplored_transitions)
                logging.critical(f"num_transitions = {num_transitions}")

                if self.last_transition_index >= num_transitions:
                    self.last_transition_index = -1

                self.last_transition_index = (self.last_transition_index + 1) % num_transitions
                target_transition = unexplored_transitions[self.last_transition_index]
                target_position, dest_map_id, distance = target_transition

                logger.debug(f"Selected transition {self.last_transition_index + 1}/{num_transitions}: {target_transition}")

                # Set frontier to the transition position on the CURRENT map (not destination map)
                # The destination map ID is unknown until we actually transition
                self.frontiers = {(current_map_id, target_position[0], target_position[1])}
                self.current_frontier = (current_map_id, target_position[0], target_position[1])
                self.steps_since_last_frontier = 0
                self.steps_towards_current_frontier = 0
                path = self.plan_path(self.get_ash_position_grid(), self.current_frontier)

                if path is None or len(path) < 2:
                    logger.critical("Cannot find a path to the transition. Excluding the frontier.")
                    self.excluded_frontiers.add(self.current_frontier)
                    self.frontiers.discard(self.current_frontier)
                    if self.current_frontier:
                        self.old_frontier = self.current_frontier
                    self.current_frontier = None
                    return False
                else: 
                    self.current_path = path[1:]
                    logger.critical(f"Moving towards unexplored map. Path: {self.current_path}")
                    return True
            else:
                logger.critical("No unexplored transitions available.")
                return False
        else:
            # Already have a path, don't need to find a new one
            return True

    def move_towards_known_area(self):
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
        current_map_id, cx, cy = current
        next_map_id, nx, ny = next_pos
        dx = nx - cx
        dy = ny - cy
        logger.debug(f"Current position: ({cx}, {cy}), Next position: ({nx}, {ny}), dx: {dx}, dy: {dy}")
        
        # Check if this is a map transition (different map IDs)
        if current_map_id != next_map_id:
            # This is a map transition - clear the path but don't reset frontiers
            logger.info(f"Map transition detected from map {current_map_id} to {next_map_id}. Clearing path.")
            self.current_path = []
            # Don't reset frontiers here - let update_slam_map handle the transition
            self.random_action_steps_remaining = 1
            return self.ppo_action
        
        if dx in [-1, 1] and dy in [-1, 1]:
            self.random_action_steps_remaining = 1
            action = self.ppo_action
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
            # Only reset frontiers if it's a truly unexpected movement on the same map
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
        map_id, x, y = position
        self.collision_counts[(map_id, x, y)] += 1

    def get_collision_count(self, position):
        map_id, x, y = position
        return self.collision_counts[(map_id, x, y)]

    def reset_collision_count(self, position):
        map_id, x, y = position
        del self.collision_counts[(map_id, x, y)]
    
    def configure_vlm(self, 
                     enabled: bool = None,
                     confidence_threshold: float = None,
                     call_frequency: int = None,
                     api_key: str = None,
                     base_url: str = None,
                     model: str = None,
                     config_file: str = None):
        """
        Configure VLM integration parameters.
        
        Args:
            enabled: Enable/disable VLM integration
            confidence_threshold: Minimum confidence for VLM decisions (0.0-1.0)
            call_frequency: How often to call VLM (every N steps)
            api_key: API key for VLM service
            base_url: Base URL for OpenAI-compatible API
            model: Model name to use
            config_file: Path to configuration file to reload
        """
        # Reload configuration if requested
        if config_file is not None:
            self.vlm_config_loader = VLMConfigLoader(config_file)
            self.load_vlm_config()
        
        if not self.vlm_config:
            logger.warning("No VLM configuration available. Cannot configure VLM.")
            return
        
        # Update configuration values
        if enabled is not None:
            self.vlm_config.enabled = enabled and VLM_AVAILABLE
            self.vlm_enabled = self.vlm_config.enabled
        
        if confidence_threshold is not None:
            self.vlm_config.decision.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        
        if call_frequency is not None:
            self.vlm_config.decision.call_frequency = max(1, call_frequency)
        
        if api_key is not None:
            self.vlm_config.api.api_key = api_key
        
        if base_url is not None:
            self.vlm_config.api.base_url = base_url
        
        if model is not None:
            self.vlm_config.api.model = model
        
        logger.info(f"VLM configured: enabled={self.vlm_config.enabled}, "
                   f"threshold={self.vlm_config.decision.confidence_threshold}, "
                   f"frequency={self.vlm_config.decision.call_frequency}, "
                   f"model={self.vlm_config.api.model}")

    def get_vlm_stats(self) -> dict:
        """
        Get statistics about VLM usage.
        
        Returns:
            dict: VLM usage statistics
        """
        # Get confidence threshold and call frequency from config if available
        confidence_threshold = None
        call_frequency = None
        if self.vlm_config:
            confidence_threshold = self.vlm_config.decision.confidence_threshold
            call_frequency = self.vlm_config.decision.call_frequency
        
        return {
            'vlm_enabled': self.vlm_enabled,
            'vlm_step_counter': getattr(self, 'vlm_step_counter', 0),
            'confidence_threshold': confidence_threshold,
            'call_frequency': call_frequency,
            'last_decision': getattr(self, 'last_vlm_decision', None),
            'context_history_length': len(getattr(self, 'vlm_context_history', deque())),
            'active_override': getattr(self, 'vlm_action_override', None) is not None,
            'override_steps_remaining': getattr(self, 'vlm_active_steps', 0)
        }
    
    def load_vlm_config(self):
        if self.vlm_config_loader:
            self.vlm_config = self.vlm_config_loader.load_config()
            if self.vlm_config:
                self.vlm_enabled = self.vlm_config.enabled
                logger.info(f"VLM configuration loaded: enabled={self.vlm_enabled}, "
                            f"threshold={self.vlm_config.decision.confidence_threshold}, "
                            f"frequency={self.vlm_config.decision.call_frequency}, "
                            f"model={self.vlm_config.api.model}")
            else:
                logger.warning("Failed to load VLM configuration.")
        else:
            logger.warning("VLM configuration loader not available.")
    
    def get_llm_action(self) -> Optional[int]:
        """
        Get an action decision from the LLM controller when no frontiers are detected.
        This method provides a synchronous interface to the async LLM controller.
        
        Returns:
            Optional[int]: Action from LLM (0-5) or None if LLM is unavailable or fails
        """
        if not self.llm_enabled or not self.llm_controller:
            print(f"[LLM] Controller not enabled (enabled={self.llm_enabled}) or not available (controller={self.llm_controller is not None})")
            logger.warning(f"LLM controller is not enabled (enabled={self.llm_enabled}) or not available (controller={self.llm_controller})")
            return None
        
        try:
            logger.info("Getting original screen image for LLM controller...")
            # Get original PyBoy screen image (160x144) instead of resized observation (120x120)
            raw_screen = self.pyboy.screen.ndarray
            if raw_screen is None:
                print("[LLM] ERROR: No screen data available from PyBoy.")
                logger.error("No screen data available from PyBoy.")
                return None
            
            # Convert BGR to RGB (PyBoy uses BGR format)
            original_screen = cv2.cvtColor(raw_screen, cv2.COLOR_BGR2RGB)
            # Ensure it's a numpy array with correct dtype
            original_screen = np.array(original_screen)[:, :, :3].astype(np.uint8)
            logger.info(f"Original screen shape: {original_screen.shape}")
            
            logger.info("Getting game stats for LLM controller...")
            # Get game stats for the LLM
            game_stats = self.llm_controller.get_game_stats(self)
            logger.info(f"Game stats: {game_stats}")
            
            # Call the async LLM method synchronously
            # Note: The OpenAI client calls inside are actually synchronous, 
            # but the method is marked async, so we use asyncio.run
            try:
                # Check if we're already in an async event loop
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in an async context, we can't use asyncio.run
                    # Create a task instead (but this requires the caller to be async)
                    # For now, log a warning and return None
                    logger.warning("LLM controller called from async context. Cannot use asyncio.run. Returning None.")
                    return None
                except RuntimeError:
                    # No running event loop, safe to use asyncio.run
                    pass
                
                logger.info("Calling LLM controller get_next_action...")
                # Run the async method synchronously - pass original screen instead of resized observation
                action = asyncio.run(self.llm_controller.get_next_action(original_screen, game_stats))
                logger.info(f"LLM controller returned action: {action}")
                return action
            except Exception as e:
                logger.error(f"Error calling LLM controller async method: {e}", exc_info=True)
                return None
                
        except Exception as e:
            logger.error(f"Error getting LLM action: {e}", exc_info=True)
            return None
    