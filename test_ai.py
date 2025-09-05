from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from ai_env import DrivingEnv
import time

# Load the trained model
model = PPO.load("driving_ai_model")

# Create environment the same way as training (vectorized)
env = make_vec_env(DrivingEnv, n_envs=1)

# Test the AI
print("Testing the trained AI...")
print("Close the game window to stop testing")

episodes = 0
while episodes < 10:  # Test for 10 episodes
    obs = env.reset()  # VecEnv returns just obs, not (obs, info)
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        # Let AI choose action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)  # VecEnv returns 4 values
        total_reward += reward[0]  # reward is an array for vectorized env
        steps += 1
        
        # Render the game (access the actual environment)
        env.envs[0].render()
        time.sleep(0.01)  # Small delay to see the action
        
        # Handle pygame events to prevent freezing
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = [True]  # VecEnv expects array
                episodes = 10  # Exit outer loop too
    
    episodes += 1
    print(f"Episode {episodes}: Total reward: {total_reward:.2f}, Steps: {steps}")

env.close()