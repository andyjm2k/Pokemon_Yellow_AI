import os
import base64
from io import BytesIO
import numpy as np
from PIL import Image
from openai import OpenAI
import logging
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMController:
    def __init__(self):
        # Initialize OpenAI client with API key from environment
        self.api_key = 'AIzaSyCWB5SsdafyjD-2-Za8HEI4F4qcbslp9XY'
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.base_url = 'Https://generativelanguage.googleapis.com/v1beta/openai'
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        # Add rate limiting parameters
        self.request_delay = 10.0  # Seconds to wait between API calls
        self.last_request_time = 0  # Track the time of the last request
        self.max_retries = 3  # Maximum number of retry attempts
        self.retry_delay = 2.0  # Base delay between retries (will increase with backoff)

    def encode_image_to_base64(self, image_array):
        """Convert numpy array to base64 string."""
        # Convert numpy array to PIL Image
        image = Image.fromarray(image_array)
        
        # Save image to bytes buffer
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        
        # Encode to base64
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def create_game_state_prompt(self, game_stats):
        # Create a prompt with the current game state.
        prompt = f"""You are an AI assistant playing Pokemon Yellow. Analyze the game state and recommend the next action.
                    Current Game State:
                    - Pokemon in Party: {game_stats.get('party_pokemon', [])}
                    - Pokemon in Box: {game_stats.get('box_count', 0)}
                    - Current Map ID: {game_stats.get('current_map', 'Unknown')}
                    - Position: {game_stats.get('position', 'Unknown')}
                    - Current Goal: {game_stats.get('current_goal', 'None')}
                    - Battle State: {game_stats.get('is_battling', False)}

                    Available actions:
                    0: A Button (Use this for interactions and confirming choices)
                    1: Right (Move right)
                    2: Left (Move left)
                    3: Up (Move up)
                    4: Down (Move down)
                    5: B Button (Use this to cancel or go back)

                    Based on the game image and current state, respond with ONLY a single number (0-5) representing the next action to take.
                    DO NOT include any other text or explanation in your response."""
        return prompt

    async def get_next_action(self, observation, game_stats):
        """Get the next action from the LLM based on the current game state."""
        # Ensure minimum delay between requests
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_delay:
            sleep_time = self.request_delay - time_since_last_request
            logger.info(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        retries = 0
        while retries <= self.max_retries:
            try:
                # Encode the observation image
                base64_image = self.encode_image_to_base64(observation)
                
                # Create the prompt with game state
                prompt = self.create_game_state_prompt(game_stats)
                
                # Update last request time before making the API call
                self.last_request_time = time.time()
                
                # Call OpenAI API with vision model
                response = self.client.chat.completions.create(
                    model="gemini-2.0-flash",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}",
                                        "detail": "low"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=10  # Reduce tokens since we only need a digit
                )

                # Parse the response
                try:
                    content = response.choices[0].message.content.strip()
                    action = int(content)
                    if 0 <= action <= 5:
                        logger.info(f"LLM suggested action: {action}")
                        return action
                    else:
                        logger.error(f"Invalid action number: {action}")
                        return None
                except (ValueError, AttributeError) as e:
                    logger.error(f"Error parsing LLM response: {e}")
                    return None

            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    logger.error(f"Max retries exceeded. Error: {e}")
                    return None
                
                # Exponential backoff
                backoff_time = self.retry_delay * (2 ** (retries - 1))
                logger.warning(f"Request failed (attempt {retries}/{self.max_retries}). Retrying in {backoff_time} seconds. Error: {e}")
                time.sleep(backoff_time)

    def get_game_stats(self, env):
        """Gather relevant game stats from the environment."""
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
            'is_battling': env.is_battling_fl
        }

    def get_pokemon_info(self, pokemon_id):
        """Get Pokemon information if it exists."""
        if pokemon_id is None:
            return None
        return pokemon_id  # You might want to expand this with more Pokemon details 