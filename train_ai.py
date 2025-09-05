from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from ai_env import DrivingEnv
import matplotlib.pyplot as plt

# Create environment
env = make_vec_env(DrivingEnv, n_envs=1)

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

print("Starting training...")
print("This will take several minutes. Watch the terminal for progress.")

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("driving_ai_model")
print("Training complete! Model saved as 'driving_ai_model'")

# Close environment
env.close()
