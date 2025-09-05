import gym
from gym import spaces
import numpy as np
from game import SimpleDrivingGame

class DrivingEnv(gym.Env):
    def __init__(self):
        super(DrivingEnv, self).__init__()
        
        self.game = SimpleDrivingGame()
        
        # Action space: 5 discrete actions
        # 0=nothing, 1=left, 2=right, 3=accelerate, 4=brake
        self.action_space = spaces.Discrete(5)
        
        # Observation space: [left_dist, right_dist, speed, angle, track_direction]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1, -1]),
            high=np.array([1, 1, 1, 1, 1]),
            dtype=np.float32
        )
        
        self.max_steps = 1000
        self.current_step = 0
        
    def seed(self, seed=None):
        """Add seed method for compatibility"""
        np.random.seed(seed)
        return [seed]
        
    def reset(self, seed=None, options=None):
        """Updated reset method"""
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        obs = self.game.reset().astype(np.float32)
        return obs, {}  # New gym format returns (obs, info)
    
    def step(self, action):
        self.current_step += 1
        
        state, reward, done = self.game.step(action)
        
        # End episode if too many steps
        if self.current_step >= self.max_steps:
            done = True
            
        # New gym format: (obs, reward, terminated, truncated, info)
        truncated = self.current_step >= self.max_steps
        terminated = done and not truncated
        
        return state.astype(np.float32), reward, terminated, truncated, {}
    
    def render(self, mode='human'):
        if mode == 'human':
            self.game.render()
    
    def close(self):
        pass