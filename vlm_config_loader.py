"""
VLM Configuration Loader for vendor-agnostic OpenAI-compatible API integration.

This module handles loading VLM configuration from:
1. YAML configuration files
2. Environment variables (override config file values)
3. Default values (fallback)

Supports any OpenAI-compatible API endpoint.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class VLMAPIConfig:
    """Configuration for VLM API connection."""
    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    model: str = "gpt-4-vision-preview"
    headers: Dict[str, str] = None
    timeout: int = 30
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
        if self.parameters is None:
            self.parameters = {
                "max_tokens": 300,
                "temperature": 0.7,
                "top_p": 1.0
            }

@dataclass
class VLMDecisionConfig:
    """Configuration for VLM decision making."""
    confidence_threshold: float = 0.75
    call_frequency: int = 10
    override_steps: int = 3
    high_confidence_threshold: float = 0.9

@dataclass
class VLMConditionsConfig:
    """Configuration for when to use VLM."""
    use_in_battle: bool = True
    use_when_stuck: bool = True
    stuck_threshold: int = 10
    use_for_goals: list = None
    use_when_really_stuck: bool = True
    really_stuck_threshold: int = 1000
    periodic_checks: bool = True
    
    def __post_init__(self):
        if self.use_for_goals is None:
            self.use_for_goals = ["red", "magenta"]

@dataclass
class VLMPromptsConfig:
    """Configuration for VLM prompts."""
    system_prompt: str = "You are an expert Pokemon Yellow player. Analyze the game screen and context to determine the best action."
    context_template: str = "Current context: Goal: {current_goal}, Battling: {is_battling}, Position: {ash_position}"
    action_guidance: str = "Available actions: 0-5. Respond in JSON format with action, confidence, and reasoning."

@dataclass
class VLMLoggingConfig:
    """Configuration for VLM logging."""
    log_decisions: bool = True
    log_api_calls: bool = False
    log_level: str = "INFO"
    save_history: bool = False
    history_file: str = "vlm_history.json"

@dataclass
class VLMConfig:
    """Complete VLM configuration."""
    enabled: bool = True
    api: VLMAPIConfig = None
    decision: VLMDecisionConfig = None
    conditions: VLMConditionsConfig = None
    prompts: VLMPromptsConfig = None
    logging: VLMLoggingConfig = None
    
    def __post_init__(self):
        if self.api is None:
            self.api = VLMAPIConfig()
        if self.decision is None:
            self.decision = VLMDecisionConfig()
        if self.conditions is None:
            self.conditions = VLMConditionsConfig()
        if self.prompts is None:
            self.prompts = VLMPromptsConfig()
        if self.logging is None:
            self.logging = VLMLoggingConfig()

class VLMConfigLoader:
    """Loads and manages VLM configuration from multiple sources."""
    
    def __init__(self, config_file: str = "vlm_config.yaml"):
        self.config_file = config_file
        self.config = VLMConfig()
    
    def load_config(self) -> VLMConfig:
        """
        Load configuration from file and environment variables.
        
        Priority (highest to lowest):
        1. Environment variables
        2. Configuration file
        3. Default values
        
        Returns:
            VLMConfig: Loaded configuration
        """
        # Start with defaults
        config_dict = asdict(VLMConfig())
        
        # Load from file if it exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config and 'vlm' in file_config:
                        config_dict = self._merge_dicts(config_dict, file_config['vlm'])
                        logger.info(f"Loaded VLM config from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading config file {self.config_file}: {e}")
        else:
            logger.warning(f"Config file {self.config_file} not found. Using defaults.")
        
        # Override with environment variables
        config_dict = self._apply_env_overrides(config_dict)
        
        # Convert to dataclass
        self.config = self._dict_to_config(config_dict)
        
        # Validate configuration
        self._validate_config()
        
        return self.config
    
    def _merge_dicts(self, base: dict, override: dict) -> dict:
        """Recursively merge dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    def _apply_env_overrides(self, config_dict: dict) -> dict:
        """Apply environment variable overrides."""
        env_mappings = {
            'VLM_ENABLED': ('enabled', lambda x: x.lower() == 'true'),
            'VLM_API_KEY': ('api.api_key', str),
            'VLM_BASE_URL': ('api.base_url', str),
            'VLM_MODEL': ('api.model', str),
            'VLM_TIMEOUT': ('api.timeout', int),
            'VLM_MAX_TOKENS': ('api.parameters.max_tokens', int),
            'VLM_TEMPERATURE': ('api.parameters.temperature', float),
            'VLM_CONFIDENCE_THRESHOLD': ('decision.confidence_threshold', float),
            'VLM_CALL_FREQUENCY': ('decision.call_frequency', int),
            'VLM_LOG_DECISIONS': ('logging.log_decisions', lambda x: x.lower() == 'true'),
            'VLM_LOG_API_CALLS': ('logging.log_api_calls', lambda x: x.lower() == 'true'),
        }
        
        for env_var, (config_path, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    converted_value = converter(env_value)
                    self._set_nested_value(config_dict, config_path, converted_value)
                    logger.info(f"Applied environment override: {env_var}={converted_value}")
                except Exception as e:
                    logger.error(f"Error converting environment variable {env_var}={env_value}: {e}")
        
        return config_dict
    
    def _set_nested_value(self, dictionary: dict, path: str, value: Any):
        """Set a value in a nested dictionary using dot notation."""
        keys = path.split('.')
        current = dictionary
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def _dict_to_config(self, config_dict: dict) -> VLMConfig:
        """Convert dictionary to VLMConfig dataclass."""
        try:
            return VLMConfig(
                enabled=config_dict.get('enabled', True),
                api=VLMAPIConfig(**config_dict.get('api', {})),
                decision=VLMDecisionConfig(**config_dict.get('decision', {})),
                conditions=VLMConditionsConfig(**config_dict.get('conditions', {})),
                prompts=VLMPromptsConfig(**config_dict.get('prompts', {})),
                logging=VLMLoggingConfig(**config_dict.get('logging', {}))
            )
        except Exception as e:
            logger.error(f"Error creating VLMConfig from dictionary: {e}")
            return VLMConfig()  # Return defaults
    
    def _validate_config(self):
        """Validate the loaded configuration."""
        config = self.config
        
        # Validate API configuration
        if config.enabled and not config.api.api_key:
            logger.warning("VLM enabled but no API key provided. VLM will be disabled.")
            config.enabled = False
        
        # Validate base URL format
        if not config.api.base_url.startswith(('http://', 'https://')):
            logger.error(f"Invalid base URL format: {config.api.base_url}")
            config.enabled = False
        
        # Validate thresholds
        if not 0.0 <= config.decision.confidence_threshold <= 1.0:
            logger.warning(f"Invalid confidence threshold: {config.decision.confidence_threshold}. Using 0.75.")
            config.decision.confidence_threshold = 0.75
        
        if not 0.0 <= config.decision.high_confidence_threshold <= 1.0:
            logger.warning(f"Invalid high confidence threshold: {config.decision.high_confidence_threshold}. Using 0.9.")
            config.decision.high_confidence_threshold = 0.9
        
        # Validate call frequency
        if config.decision.call_frequency < 1:
            logger.warning(f"Invalid call frequency: {config.decision.call_frequency}. Using 10.")
            config.decision.call_frequency = 10
    
    def save_config(self, filepath: str = None):
        """Save current configuration to file."""
        if filepath is None:
            filepath = self.config_file
        
        config_dict = {'vlm': asdict(self.config)}
        
        try:
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            logger.info(f"Saved VLM config to {filepath}")
        except Exception as e:
            logger.error(f"Error saving config to {filepath}: {e}")
    
    def get_api_headers(self) -> Dict[str, str]:
        """Get complete headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            **self.config.api.headers
        }
        
        # Add authorization header if API key is provided
        if self.config.api.api_key:
            headers["Authorization"] = f"Bearer {self.config.api.api_key}"
        
        return headers
    
    def get_api_endpoint(self) -> str:
        """Get the complete API endpoint URL."""
        base_url = self.config.api.base_url.rstrip('/')
        if not base_url.endswith('/v1'):
            base_url += '/v1'
        return f"{base_url}/chat/completions"
    
    def print_config_summary(self):
        """Print a summary of the current configuration."""
        config = self.config
        print("VLM Configuration Summary:")
        print(f"  Enabled: {config.enabled}")
        print(f"  API Base URL: {config.api.base_url}")
        print(f"  Model: {config.api.model}")
        print(f"  API Key: {'Set' if config.api.api_key else 'Not set'}")
        print(f"  Confidence Threshold: {config.decision.confidence_threshold}")
        print(f"  Call Frequency: {config.decision.call_frequency}")
        print(f"  Use in Battle: {config.conditions.use_in_battle}")
        print(f"  Log Decisions: {config.logging.log_decisions}")

def load_vlm_config(config_file: str = "vlm_config.yaml") -> VLMConfig:
    """Convenience function to load VLM configuration."""
    loader = VLMConfigLoader(config_file)
    return loader.load_config()

# Example usage and testing
if __name__ == "__main__":
    # Test the configuration loader
    loader = VLMConfigLoader()
    config = loader.load_config()
    loader.print_config_summary()
    
    # Test saving configuration
    loader.save_config("test_vlm_config.yaml")
    print("\nTest configuration saved to test_vlm_config.yaml") 