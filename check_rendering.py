from ai_env import DrivingEnv
import time

# Create environment directly (not vectorized)
env = DrivingEnv()

print("Testing if car movement is visible...")

# Reset environment
obs, _ = env.reset()

# Force the car to accelerate for 50 steps
for i in range(50):
    action = 3  # Force accelerate action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(f"Step {i}: Speed = {env.game.car_speed:.2f}, Position = ({env.game.car_x:.1f}, {env.game.car_y:.1f})")
    time.sleep(0.1)
    
    if terminated or truncated:
        print("Episode ended!")
        break

env.close()