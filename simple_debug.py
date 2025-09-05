from ai_env import CheckpointRacingEnv
import time

# Test with random actions first to see if the environment works
env = CheckpointRacingEnv()

print("Testing checkpoint racing environment with random actions...")
print("This will help us see if the basic system works.")

try:
    obs = env.reset()
    print(f"Initial state: {obs}")
    print("State format: [distance_to_checkpoint, angle_to_checkpoint, speed, car_angle, progress]")
    
    for step in range(50):
        # Random action
        action = env.action_space.sample()
        action_names = ["nothing", "left", "right", "accelerate", "brake"]
        
        print(f"\nStep {step}:")
        print(f"  Action: {action} ({action_names[action]})")
        print(f"  Current checkpoint: {env.game.current_checkpoint}")
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        print(f"  New state: {obs}")
        print(f"  Reward: {reward:.2f}")
        
        if reward > 50:
            print(f"  *** CHECKPOINT REACHED! ***")
        
        # Render
        env.render()
        time.sleep(0.3)
        
        if done:
            print(f"  Episode ended after {step+1} steps")
            break
        
        # Handle pygame events
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
        
        if done:
            break
    
    print("Environment test completed successfully!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    env.close()