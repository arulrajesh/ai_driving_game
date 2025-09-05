from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from ai_env import CheckpointRacingEnv  # Updated import
import time

# Load the trained model
model = PPO.load("checkpoint_racing_model")  # Updated model name

# Create environment the same way as training (vectorized)
env = make_vec_env(CheckpointRacingEnv, n_envs=1)  # Updated class name

# Test the AI
print("Testing the trained checkpoint racing AI...")
print("Watch the AI navigate between checkpoints!")
print("Green circle = current target checkpoint")
print("Close the game window to stop testing")

episodes = 0
while episodes < 5:  # Test for 5 episodes
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    checkpoints_reached = 0
    
    while not done:
        # Let AI choose action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        steps += 1
        
        # Count checkpoints reached (big rewards indicate checkpoint)
        if reward[0] > 50:
            checkpoints_reached += 1
            print(f"  Checkpoint {checkpoints_reached} reached!")
        
        # Render the game
        env.envs[0].render()
        time.sleep(0.02)  # Small delay to see the action
        
        # Handle pygame events to prevent freezing
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = [True]
                episodes = 5  # Exit outer loop too
    
    episodes += 1
    
    # Check if lap was completed
    laps_completed = env.envs[0].game.laps_completed
    if laps_completed > 0:
        print(f"🎉 Episode {episodes}: LAP COMPLETED! Total reward: {total_reward:.2f}, Steps: {steps}, Checkpoints: {checkpoints_reached}")
    else:
        print(f"Episode {episodes}: Total reward: {total_reward:.2f}, Steps: {steps}, Checkpoints: {checkpoints_reached}")

env.close()
print("Testing complete!")