"""
Example usage of vendor-agnostic VLM integration with Pokemon Yellow environment.

This example shows how to:
1. Initialize the environment with VLM capabilities
2. Configure VLM parameters for different API providers
3. Run the environment with VLM decision making
4. Monitor VLM usage statistics

Supports any OpenAI-compatible API endpoint including:
- OpenAI (GPT-4V, GPT-4o)
- Local models (Ollama, LLaMA, etc.)
- Other cloud providers (Together AI, Anthropic via proxy, etc.)
"""

import os
import time
from environment_pyboy_neat_pkmn_yellow_goal_pf import GbaGame

def demo_openai_gpt4v():
    """Demonstrate using OpenAI GPT-4V"""
    print("=== OpenAI GPT-4V Demo ===")
    
    env = GbaGame(max_episodes=1000)
    
    # Configure for OpenAI GPT-4V
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        env.configure_vlm(
            enabled=True,
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            model="gpt-4-vision-preview",
            confidence_threshold=0.75,
            call_frequency=10
        )
        print("✅ OpenAI GPT-4V configured successfully!")
        return env
    else:
        print("❌ No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        return None

def demo_local_ollama():
    """Demonstrate using local Ollama with LLaVA"""
    print("\n=== Local Ollama (LLaVA) Demo ===")
    
    env = GbaGame(max_episodes=1000)
    
    # Configure for local Ollama
    env.configure_vlm(
        enabled=True,
        base_url="http://localhost:11434/v1",
        model="llava:latest",  # or "bakllava:latest"
        confidence_threshold=0.6,  # Lower threshold for local models
        call_frequency=15
    )
    print("✅ Local Ollama configured! Make sure Ollama is running with LLaVA model.")
    return env

def demo_together_ai():
    """Demonstrate using Together AI"""
    print("\n=== Together AI Demo ===")
    
    env = GbaGame(max_episodes=1000)
    
    # Configure for Together AI
    api_key = os.getenv('TOGETHER_API_KEY')
    if api_key:
        env.configure_vlm(
            enabled=True,
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            model="meta-llama/Llama-Vision-Free",  # Check Together AI for available models
            confidence_threshold=0.7,
            call_frequency=12
        )
        print("✅ Together AI configured successfully!")
        return env
    else:
        print("❌ No Together AI API key found. Set TOGETHER_API_KEY environment variable.")
        return None

def demo_config_file():
    """Demonstrate using a configuration file"""
    print("\n=== Configuration File Demo ===")
    
    # Create a sample config file
    config_content = """
vlm:
  enabled: true
  
  api:
    base_url: "https://api.openai.com/v1"
    api_key: null  # Will be overridden by environment variable
    model: "gpt-4o"
    timeout: 30
    parameters:
      max_tokens: 250
      temperature: 0.5
  
  decision:
    confidence_threshold: 0.8
    call_frequency: 8
    override_steps: 2
    high_confidence_threshold: 0.95
  
  conditions:
    use_in_battle: true
    use_when_stuck: true
    stuck_threshold: 5
    use_for_goals: ["red", "magenta", "green"]
    periodic_checks: true
  
  prompts:
    system_prompt: |
      You are an expert Pokemon Yellow speedrunner. Make optimal decisions to progress quickly.
    
    context_template: |
      Game State:
      - Current Goal: {current_goal}
      - Battle Status: {is_battling}
      - Position: {ash_position}
      - HP: {ash_hp}
      - Stuck for {steps_at_same_position} steps
      - Last Action: {last_action}
      - Has Path: {has_path}
    
    action_guidance: |
      Actions: 0=A(interact), 1=Right, 2=Left, 3=Up, 4=Down, 5=B(cancel)
      
      JSON Response Format:
      {"action": <0-5>, "confidence": <0.0-1.0>, "reasoning": "<explanation>"}
      
      Priority Guidelines:
      1. Battles: Press A to attack/select moves
      2. Stuck: Try different movement directions
      3. Menus: A to confirm, B to cancel
      4. Exploration: Move toward objectives

logging:
  log_decisions: true
  log_api_calls: false
"""
    
    # Write config file
    with open("demo_vlm_config.yaml", "w") as f:
        f.write(config_content)
    
    # Initialize environment with config file
    env = GbaGame(max_episodes=1000)
    env.configure_vlm(config_file="demo_vlm_config.yaml")
    
    print("✅ Configuration loaded from demo_vlm_config.yaml!")
    return env

def run_environment_demo(env, demo_name, steps=200):
    """Run a demo with the configured environment"""
    if not env:
        return
    
    print(f"\n--- Running {demo_name} ---")
    
    observation, info = env.reset()
    done = False
    step_count = 0
    vlm_usage_count = 0
    
    while not done and step_count < steps:
        # In a real scenario, this action would come from your RL agent
        action = env.action_space.sample()  # Random action for demo
        
        # Take step in environment (VLM may override the action)
        observation, reward, done, truncated, info = env.step(action)
        step_count += 1
        
        # Check VLM statistics every 25 steps
        if step_count % 25 == 0:
            stats = env.get_vlm_stats()
            if stats['vlm_step_counter'] > vlm_usage_count:
                vlm_usage_count = stats['vlm_step_counter']
                print(f"Step {step_count}: VLM consulted {vlm_usage_count} times")
                if stats['last_decision']:
                    decision = stats['last_decision']
                    print(f"  Last decision: action={decision['action']}, "
                          f"confidence={decision['confidence']:.2f}")
        
        # Small delay for visualization
        time.sleep(0.01)
    
    # Final statistics
    final_stats = env.get_vlm_stats()
    print(f"{demo_name} Results:")
    print(f"  Steps: {step_count}")
    print(f"  VLM consultations: {final_stats['vlm_step_counter']}")
    print(f"  VLM usage rate: {final_stats['vlm_step_counter']/step_count*100:.1f}%")
    
    env.close()

def main():
    print("Pokemon Yellow Vendor-Agnostic VLM Integration Demo")
    print("=" * 55)
    
    # Show available demos
    demos = [
        ("OpenAI GPT-4V", demo_openai_gpt4v),
        ("Local Ollama", demo_local_ollama),
        ("Together AI", demo_together_ai),
        ("Config File", demo_config_file)
    ]
    
    print("\nAvailable VLM providers:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"{i}. {name}")
    
    # Run all available demos
    print("\n" + "=" * 55)
    for demo_name, demo_func in demos:
        try:
            env = demo_func()
            if env:
                run_environment_demo(env, demo_name, steps=100)
        except Exception as e:
            print(f"❌ Error in {demo_name} demo: {e}")
    
    # Clean up demo files
    try:
        os.remove("demo_vlm_config.yaml")
        print("\n🧹 Cleaned up demo files.")
    except:
        pass

def show_provider_examples():
    """Show configuration examples for different providers"""
    print("\n" + "=" * 55)
    print("Configuration Examples for Different Providers")
    print("=" * 55)
    
    examples = {
        "OpenAI GPT-4V": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4-vision-preview",
            "api_key": "sk-xxx...",
            "notes": "Premium service, excellent vision capabilities"
        },
        "OpenAI GPT-4o": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o",
            "api_key": "sk-xxx...",
            "notes": "Faster and cheaper than GPT-4V"
        },
        "Local Ollama (LLaVA)": {
            "base_url": "http://localhost:11434/v1",
            "model": "llava:latest",
            "api_key": "not_required",
            "notes": "Free local inference, requires setup"
        },
        "Together AI": {
            "base_url": "https://api.together.xyz/v1",
            "model": "meta-llama/Llama-Vision-Free",
            "api_key": "xxx...",
            "notes": "Good performance, competitive pricing"
        },
        "Azure OpenAI": {
            "base_url": "https://your-resource.openai.azure.com/openai/deployments/your-deployment",
            "model": "gpt-4-vision",
            "api_key": "xxx...",
            "notes": "Enterprise OpenAI through Azure"
        }
    }
    
    for provider, config in examples.items():
        print(f"\n{provider}:")
        print(f"  base_url: {config['base_url']}")
        print(f"  model: {config['model']}")
        print(f"  api_key: {config['api_key']}")
        print(f"  notes: {config['notes']}")

if __name__ == "__main__":
    # Show provider examples first
    show_provider_examples()
    
    # Ask user if they want to run demos
    response = input("\nRun live demos? (requires API keys/local setup) [y/N]: ")
    if response.lower() == 'y':
        main()
    else:
        print("Demo skipped. Configure your API keys and endpoints to run live demos.") 