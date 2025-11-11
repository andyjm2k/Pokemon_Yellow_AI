import os
import base64
from io import BytesIO
import numpy as np
from PIL import Image
from openai import OpenAI
import logging
import json
import time
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMController:
    def __init__(self):
        # Initialize OpenAI client with API key from environment
        self.api_key = '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'
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.base_url = 'http://192.168.1.109:1234/v1'
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        # Set default image size for optimization
        self.target_image_size = (160, 144)  # Default GBA resolution but can be reduced further
        self.image_format = "JPEG"  # More efficient than PNG for this use case
        self.image_quality = 80  # Adjust quality (0-100) to balance size and clarity
        
        # Image caching to avoid redundant processing
        self.image_cache = {}
        self.cache_size_limit = 100  # Limit cache size to prevent memory issues

    def compute_image_hash(self, image_array):
        """Compute a hash of the image for caching purposes."""
        # Convert to bytes and compute MD5 hash
        image_bytes = image_array.tobytes()
        return hashlib.md5(image_bytes).hexdigest()

    def resize_image(self, image_array):
        """Resize image to target dimensions for faster processing, or return original if already correct size."""
        image = Image.fromarray(image_array)
        # Check if image is already at target size - if so, return original to preserve quality
        if image.size == self.target_image_size:
            return image
        # Otherwise resize to target dimensions
        resized_image = image.resize(self.target_image_size, Image.LANCZOS)
        return resized_image

    def encode_image_to_base64(self, image_array):
        """Convert numpy array to base64 string with optimization and caching."""
        # Check if we've already processed this image
        image_hash = self.compute_image_hash(image_array)
        if image_hash in self.image_cache:
            logger.debug("Using cached image encoding")
            return self.image_cache[image_hash]
        
        # First resize the image
        image = self.resize_image(image_array)
        
        # Save image to bytes buffer with compression
        buffered = BytesIO()
        image.save(buffered, format=self.image_format, quality=self.image_quality, optimize=True)
        
        # Encode to base64
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Cache the result
        if len(self.image_cache) >= self.cache_size_limit:
            # Remove a random item if cache is full
            self.image_cache.pop(next(iter(self.image_cache)))
        self.image_cache[image_hash] = img_str
        
        return img_str

    def create_game_state_prompt(self, game_stats):
        # Improved prompt with simplified, clear, and structured instructions.
        # Include recent action history for context (last 20 actions)
        llm_history = game_stats.get('llm_action_history', [])
        history_text = ""
        if llm_history:
            history_text = f"\nRecent Action History (last {len(llm_history)} of up to 20 actions):\n"
            for i, action_desc in enumerate(llm_history, 1):
                history_text += f"  {i}. {action_desc}\n"
            history_text += "Consider this history when deciding your next action. Look for patterns and avoid getting stuck in loops.\n"
        else:
            history_text = "\n(No previous actions recorded yet)\n"
        
        prompt = f"""
Think hard. Take your time. You are the greatest pokemon trainer in the world. You are controlling a Pokémon Yellow game on the gameboy color. 
Analyze the image, then choose the best action from the options below.

Game State:
- Current Map ID: {game_stats.get('current_map', 'Unknown')}
- Player Position: {game_stats.get('position', 'Unknown')}
- Previous Position: {game_stats.get('last_position', 'Unknown')}
- Last Action: {game_stats.get('last_action', 'None')}
{history_text}
Available Actions:
0: A Button (Interact/confirm)
1: Move Right
2: Move Left
3: Move Up
4: Move Down
5: B Button (Cancel/back)

Rules:
1. Your response must be only a SINGLE digit for the action you want to take.
2. Identify and avoid obstacles in the game which include trees, rocks, stumps, mountains, fences, and other obstacles.
3. Explore the maps looking for new areas to discover.
4. Avoid repeating the last action if it previously failed unless necessary.
5. If you are stuck, try different movement directions (1-4).
6. If you are in a menu, use A to select, B to back out.
7. If you are in a battle, use A to select moves/attacks, B to back out.
8. Consider your recent action history - if you've been repeating the same actions, try a different approach.

What is your decision? Respond with a single sentence explaining your decision and the SINGLE digit for the action you want to take.
Example: "I will move right to avoid the obstacle Action: 1"
Example: "I will move up to explore the new area Action: 3"
Example: "I will move down to avoid the obstacle Action: 4"
Example: "I will move left to avoid the obstacle Action: 2"
Example: "I will use B to back out of the menu Action: 5"
        """
        return prompt

    async def get_next_action(self, observation, game_stats):
        logger.info(f"Game Stats: {game_stats}")
        """Get the next action from the LLM based on the current game state."""
        try:
            # Validate observation
            if observation is None:
                logger.error("Observation is None - cannot process LLM request")
                print("[LLM] ERROR: Observation is None")
                return None
            
            # Check if observation is a numpy array and has data
            if hasattr(observation, 'size'):
                if observation.size == 0:
                    logger.error("Observation is empty - cannot process LLM request")
                    print("[LLM] ERROR: Observation is empty")
                    return None
            elif hasattr(observation, '__len__'):
                if len(observation) == 0:
                    logger.error("Observation is empty - cannot process LLM request")
                    print("[LLM] ERROR: Observation is empty")
                    return None
            
            # Encode the observation image
            base64_image = self.encode_image_to_base64(observation)
            
            # Create the prompt with game state
            prompt = self.create_game_state_prompt(game_stats)
            
            # Log summary of what's being sent (using original resolution)
            print(f"[LLM] Request: Original screen image ({observation.shape}), Prompt ({len(prompt)} chars), Map: {game_stats.get('current_map')}, Pos: {game_stats.get('position')}, Goal: {game_stats.get('current_goal')}")
            
            # Set temperature lower for more deterministic responses
            # and use lower max_tokens since we only need a single digit
            response = self.client.chat.completions.create(
                model="qwen/qwen3-vl-30b",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{self.image_format.lower()};base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=50,  # We only need a single digit response
                temperature=0.8  # Lower temperature for more consistent responses
            )
            # Parse the response
            try:
                if len(response.choices) > 0:
                    content = response.choices[0].message.content
                    if content:
                        content = content.strip()
                        print(f"[LLM] Raw response: {content}")
                        
                        # Try to extract action number from the response
                        # Look for "Action: X" pattern first
                        import re
                        action_match = re.search(r'Action:\s*(\d)', content, re.IGNORECASE)
                        if action_match:
                            action = int(action_match.group(1))
                        else:
                            # Fallback: look for a single digit at the end or anywhere
                            digits = re.findall(r'\d', content)
                            if digits:
                                action = int(digits[-1])  # Take the last digit found
                            else:
                                # Last resort: try to parse the whole content as an integer
                                action = int(content)
                        
                        if 0 <= action <= 5:
                            action_names = {0: "A Button", 1: "Right", 2: "Left", 3: "Up", 4: "Down", 5: "B Button"}
                            print(f"[LLM] Response: Action {action} ({action_names.get(action, 'Unknown')})")
                            logger.info(f"LLM suggested action: {action}")
                            return action
                        else:
                            logger.error(f"Invalid action number: {action}")
                            print(f"[LLM] ERROR: Action {action} is out of valid range (0-5)")
                            return None
                    else:
                        logger.error("Response content is empty")
                        print("[LLM] ERROR: Response content is empty")
                        return None
                else:
                    logger.error("No choices in response")
                    print("[LLM] ERROR: No choices in response")
                    return None
            except (ValueError, AttributeError) as e:
                logger.error(f"Error parsing LLM response: {e}")
                print(f"[LLM] ERROR parsing response: {e}")
                import traceback
                print(f"[LLM] Traceback: {traceback.format_exc()}")
                return None

        except Exception as e:
            logger.error(f"Error getting next action from LLM: {e}")
            print(f"[LLM] ERROR in get_next_action: {e}")
            import traceback
            print(f"[LLM] Traceback: {traceback.format_exc()}")
            return None

    def get_game_stats(self, env):
        """Gather relevant game stats from the environment."""
        # Get LLM action history (last 20 actions)
        llm_history = list(env.llm_action_history) if hasattr(env, 'llm_action_history') else []
        action_names = {0: "A Button", 1: "Right", 2: "Left", 3: "Up", 4: "Down", 5: "B Button"}
        llm_history_named = [f"{action} ({action_names.get(action, 'Unknown')})" for action in llm_history]
        
        return {
            'party_pokemon': [
                self.get_pokemon_info(env.pkm_one),
                self.get_pokemon_info(env.pkm_two),
                self.get_pokemon_info(env.pkm_three),
                self.get_pokemon_info(env.pkm_four),
                self.get_pokemon_info(env.pkm_five),
                self.get_pokemon_info(env.pkm_six)
            ],
            'box_count': env.current_box,
            'current_map': env.current_map_id,
            'position': env.ash_position,
            'current_goal': env.global_goal,
            'is_battling': env.is_battling_fl,
            'last_action': env.last_action,
            'last_position': env.ash_old_position,
            'llm_action_history': llm_history_named  # Last 20 LLM actions with names
        }

    def get_pokemon_info(self, pokemon_id):
        """Get Pokemon information if it exists."""
        if pokemon_id is None:
            return None
        return pokemon_id  # You might want to expand this with more Pokemon details 