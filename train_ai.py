from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from ai_env import CheckpointRacingEnv
import matplotlib.pyplot as plt

# Create environment
env = make_vec_env(CheckpointRacingEnv, n_envs=1)

# Create the AI model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    device="cpu"  # Use "cuda" if you have GPU
)

print("Starting checkpoint racing training...")
print("The AI needs to learn to navigate between checkpoints in sequence.")
print("Look for 'ep_rew_mean' to increase as the AI learns to reach more checkpoints.")

# Train the model
model.learn(total_timesteps=150000)  # More training time for complex task

# Save the trained model
model.save("checkpoint_racing_model")
print("Training complete! Model saved as 'checkpoint_racing_model'")

# Close environment
env.close()