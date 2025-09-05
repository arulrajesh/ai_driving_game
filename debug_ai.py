from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from ai_env import DrivingEnv
import time

# Load the trained model
model = PPO.load("driving_ai_model")

# Create environment
env = make_vec_env(DrivingEnv, n_envs=1)

print("Debugging AI actions...")
print("Actions: 0=nothing, 1=left, 2=right, 3=accelerate, 4=brake")

obs = env.reset()
done = False
step_count = 0

while not done and step_count < 100:  # Just test first 100 steps
    # Let AI choose action
    action, _ = model.predict(obs, deterministic=True)
    
    # Print what the AI is doing
    action_names = ["nothing", "left", "right", "accelerate", "brake"]
    print(f"Step {step_count}: AI chose action {action[0]} ({action_names[action[0]]})")
    print(f"  Current state: {obs[0]}")
    
    obs, reward, done, info = env.step(action)
    
    # Render the game
    env.envs[0].render()
    time.sleep(0.5)  # Slower so we can see what's happening
    
    step_count += 1
    
    # Handle pygame events
    import pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = [True]

env.close()
print("Debug complete!")