from stable_baselines3 import PPO
from ai_env import CheckpointRacingEnv
import time

# Check if model exists first
try:
    model = PPO.load("checkpoint_racing_model")
    print("Model loaded successfully!")
except:
    print("No trained model found. Train first with: python train_ai.py")
    exit()

# Create environment directly (not vectorized)
env = CheckpointRacingEnv()

print("Debugging trained AI on checkpoint gates racing...")
print("Actions: 0=nothing, 1=left, 2=right, 3=accelerate, 4=brake")
print("State: [distance_to_gate, angle_to_gate, speed, alignment, progress]")

# Handle both reset formats
try:
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result
except Exception as e:
    print(f"Reset error: {e}")
    exit()

done = False
step_count = 0
total_reward = 0

while not done and step_count < 150:  # Debug first 150 steps
    try:
        # Let AI choose action
        action, _ = model.predict(obs, deterministic=True)
        
        # Print what the AI is doing
        action_names = ["nothing", "left", "right", "accelerate", "brake"]
        print(f"\nStep {step_count}:")
        print(f"  Action: {action} ({action_names[action]})")
        print(f"  State: {obs}")
        print(f"  Current gate: {env.game.current_gate + 1}/{env.game.total_gates}")
        print(f"  Gates passed: {env.game.gates_passed}")
        print(f"  Distance to gate: {obs[0]:.3f}")
        print(f"  Angle to gate: {obs[1]:.3f} ({'left' if obs[1] < -0.2 else 'right' if obs[1] > 0.2 else 'straight'})")
        print(f"  Car speed: {obs[2]:.3f}")
        print(f"  Aligned with gate: {'Yes' if obs[3] > 0.5 else 'No'}")
        
        # Take step
        step_result = env.step(action)
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
        total_reward += reward
        
        # Check for gate passage
        if reward > 50:
            print(f"  🎉 GATE {env.game.gates_passed} PASSED! Reward: {reward:.1f}")
            print(f"  Next target: Gate {env.game.current_gate + 1}")
        
        # Render the game
        env.render()
        time.sleep(0.4)  # Slower so we can see what's happening
        
        step_count += 1
        
        # Handle pygame events
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                
    except Exception as e:
        print(f"Error during step {step_count}: {e}")
        break

env.close()
print(f"\n=== DEBUG COMPLETE ===")
print(f"Total reward: {total_reward:.2f}")
print(f"Steps taken: {step_count}")
print(f"Gates passed: {env.game.gates_passed} out of {env.game.total_gates}")

if env.game.gates_passed >= env.game.total_gates:
    print("🏆 AI COMPLETED ALL GATES!")
elif env.game.gates_passed > 0:
    print(f"✅ AI passed {env.game.gates_passed} gates - partial success!")
else:
    print("❌ AI didn't pass any gates - needs more training")

# Performance assessment
success_rate = env.game.gates_passed / env.game.total_gates * 100
print(f"Success rate: {success_rate:.1f}%")

if success_rate == 100:
    print("🌟 Perfect performance!")
elif success_rate > 50:
    print("🚗 Good driving skills!")
elif success_rate > 20:
    print("🔧 Some progress, but needs improvement")
else:
    print("🚧 Needs significant training")