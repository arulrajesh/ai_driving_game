import gym
from gym import spaces
import numpy as np
from game import CheckpointGatesGame

class CheckpointRacingEnv(gym.Env):
    def __init__(self):
        super(CheckpointRacingEnv, self).__init__()
        
        self.game = CheckpointGatesGame()
        
        # Action space: 5 discrete actions
        # 0=nothing, 1=left, 2=right, 3=accelerate, 4=brake
        self.action_space = spaces.Discrete(5)
        
        # Observation space: [distance_to_checkpoint, angle_to_checkpoint, speed, car_angle, progress]
        self.observation_space = spaces.Box(
            low=np.array([0, -1, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, 1]),
            dtype=np.float32
        )
        
        self.max_steps = 3000  # Enough time to complete the course
        self.current_step = 0
        
    def seed(self, seed=None):
        """Add seed method for compatibility"""
        np.random.seed(seed)
        return [seed]
        
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        obs = self.game.reset().astype(np.float32)
        
        # Return format depends on gym version
        # Try both formats for compatibility
        try:
            # New format (obs, info)
            return obs, {}
        except:
            # Old format (just obs)
            return obs
    
    def step(self, action):
        self.current_step += 1
        
        state, reward, done = self.game.step(action)
        
        # End episode if max steps reached
        if self.current_step >= self.max_steps:
            done = True
        
        truncated = self.current_step >= self.max_steps
        terminated = done and not truncated
        
        return state.astype(np.float32), reward, terminated, truncated, {}
    
    def render(self, mode='human'):
        if mode == 'human':
            self.game.render()
    
    def close(self):
        pass