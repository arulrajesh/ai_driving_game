import pygame
import math
import numpy as np

class CheckpointGatesGame:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Checkpoint Gates Racing AI")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        
        # Car properties
        self.car_x = width // 2
        self.car_y = height - 100
        self.car_angle = 0
        self.car_speed = 0
        self.max_speed = 5
        self.car_width = 15
        self.car_height = 25
        
        # Checkpoint gates - narrow passages the car MUST drive through
        self.gates = [
            # Each gate is defined by two points (left post, right post)
            ((width//2 - 30, height - 200), (width//2 + 30, height - 200)),  # Gate 1
            ((width//2 + 150, height - 350), (width//2 + 210, height - 350)), # Gate 2 (right)
            ((width//2 + 150, height//2), (width//2 + 210, height//2)),       # Gate 3 (right middle)
            ((width//2 - 30, height//2), (width//2 + 30, height//2)),         # Gate 4 (center)
            ((width//2 - 210, height//2), (width//2 - 150, height//2)),       # Gate 5 (left)
            ((width//2 - 210, height - 350), (width//2 - 150, height - 350)), # Gate 6 (left high)
            ((width//2 - 30, height - 450), (width//2 + 30, height - 450)),   # Gate 7 (finish)
        ]
        
        self.current_gate = 0
        self.total_gates = len(self.gates)
        self.gates_passed = 0
        
        # Game state
        self.episode_reward = 0
        self.steps_taken = 0
        self.max_steps = 1000
        
    def reset(self):
        """Reset the game to initial state"""
        self.car_x = self.width // 2
        self.car_y = self.height - 50  # Start at bottom
        self.car_angle = 0
        self.car_speed = 0
        self.current_gate = 0
        self.gates_passed = 0
        self.episode_reward = 0
        self.steps_taken = 0
        
        # Reset distance tracking
        if hasattr(self, 'prev_distance_to_gate'):
            delattr(self, 'prev_distance_to_gate')
            
        return self.get_state()
    
    def get_state(self):
        """Get current game state for AI"""
        if self.current_gate >= len(self.gates):
            # All gates passed
            gate_left = gate_right = (self.width//2, 0)
        else:
            gate_left, gate_right = self.gates[self.current_gate]
        
        # Gate center
        gate_center_x = (gate_left[0] + gate_right[0]) / 2
        gate_center_y = (gate_left[1] + gate_right[1]) / 2
        
        # Vector to gate center
        dx = gate_center_x - self.car_x
        dy = gate_center_y - self.car_y
        distance_to_gate = math.sqrt(dx*dx + dy*dy)
        
        # Angle to gate (relative to car's current heading)
        angle_to_gate = math.atan2(dx, -dy)  # -dy because y increases downward
        car_angle_rad = math.radians(self.car_angle)
        relative_angle = angle_to_gate - car_angle_rad
        
        # Normalize angle to [-pi, pi]
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi
        
        # Check if car is aligned with gate (facing the right direction)
        alignment = abs(relative_angle) < math.pi/4  # Within 45 degrees
        
        # Normalize values for AI
        normalized_distance = min(distance_to_gate / 400.0, 1.0)
        normalized_angle = relative_angle / math.pi
        normalized_speed = self.car_speed / self.max_speed
        
        return np.array([
            normalized_distance,    # Distance to next gate
            normalized_angle,       # Angle to next gate
            normalized_speed,       # Current speed
            float(alignment),       # Whether pointing toward gate (0 or 1)
            self.gates_passed / self.total_gates  # Progress
        ])
    
    def step(self, action):
        """Execute one game step based on action"""
        self.steps_taken += 1
        
        # Actions: 0=nothing, 1=left, 2=right, 3=accelerate, 4=brake
        if action == 1:  # Turn left
            self.car_angle -= 6
        elif action == 2:  # Turn right
            self.car_angle += 6
        elif action == 3:  # Accelerate
            self.car_speed = min(self.car_speed + 0.8, self.max_speed)
        elif action == 4:  # Brake
            self.car_speed = max(self.car_speed - 1.5, 0)
        
        # Natural deceleration
        self.car_speed *= 0.98
        
        # Move car
        angle_rad = math.radians(self.car_angle)
        self.car_x += self.car_speed * math.sin(angle_rad)
        self.car_y -= self.car_speed * math.cos(angle_rad)
        
        # Keep car on screen with soft boundaries
        margin = 30
        if self.car_x < margin:
            self.car_x = margin
        elif self.car_x > self.width - margin:
            self.car_x = self.width - margin
        
        if self.car_y < margin:
            self.car_y = margin
        elif self.car_y > self.height - margin:
            self.car_y = self.height - margin
        
        # Check if passed through current gate
        gate_passed = self.check_gate_passage()
        
        # Calculate reward
        reward = self.calculate_reward(gate_passed)
        self.episode_reward += reward
        
        # Episode ends if all gates passed or time limit
        done = (self.gates_passed >= self.total_gates or 
                self.steps_taken >= self.max_steps)
        
        return self.get_state(), reward, done
    
    def check_gate_passage(self):
        """Check if car passed through the current gate"""
        if self.current_gate >= len(self.gates):
            return False
        
        gate_left, gate_right = self.gates[self.current_gate]
        
        # Check if car is at the gate's Y level (within tolerance)
        gate_y = gate_left[1]  # Both posts have same Y
        if abs(self.car_y - gate_y) < 20:  # Within 20 pixels of gate line
            
            # Check if car X is between the gate posts
            gate_left_x = gate_left[0]
            gate_right_x = gate_right[0]
            
            if gate_left_x <= self.car_x <= gate_right_x:
                # Passed through gate!
                self.current_gate += 1
                self.gates_passed += 1
                return True
        
        return False
    
    def calculate_reward(self, gate_passed):
        """Ultra-simple reward: only reward gate passage"""
        
        if gate_passed:
            # MASSIVE reward for passing through gate
            return 100
        
        # Small negative reward per step to encourage efficiency
        return -0.1
    
    def render(self):
        """Render the game"""
        self.screen.fill(self.BLACK)
        
        # Draw all gates
        for i, (gate_left, gate_right) in enumerate(self.gates):
            if i == self.current_gate:
                # Current target gate - bright green
                color = self.GREEN
                thickness = 8
            elif i < self.current_gate:
                # Passed gates - gray
                color = self.GRAY
                thickness = 4
            else:
                # Future gates - white
                color = self.WHITE
                thickness = 4
            
            # Draw gate posts
            pygame.draw.circle(self.screen, color, gate_left, 10, thickness//2)
            pygame.draw.circle(self.screen, color, gate_right, 10, thickness//2)
            
            # Draw gate line
            pygame.draw.line(self.screen, color, gate_left, gate_right, thickness//2)
            
            # Draw gate number
            font = pygame.font.Font(None, 24)
            text = font.render(str(i+1), True, color)
            gate_center_x = (gate_left[0] + gate_right[0]) // 2
            gate_center_y = (gate_left[1] + gate_right[1]) // 2
            self.screen.blit(text, (gate_center_x - 6, gate_center_y - 30))
        
        # Draw car
        car_points = self.get_car_corners()
        pygame.draw.polygon(self.screen, self.RED, car_points)
        
        # Draw direction line
        car_front_x = self.car_x + 20 * math.sin(math.radians(self.car_angle))
        car_front_y = self.car_y - 20 * math.cos(math.radians(self.car_angle))
        pygame.draw.line(self.screen, self.YELLOW, (self.car_x, self.car_y), (car_front_x, car_front_y), 3)
        
        # Draw info
        font = pygame.font.Font(None, 36)
        speed_text = font.render(f"Speed: {self.car_speed:.1f}", True, self.WHITE)
        gate_text = font.render(f"Gate: {self.current_gate+1}/{self.total_gates}", True, self.WHITE)
        reward_text = font.render(f"Reward: {self.episode_reward:.1f}", True, self.WHITE)
        
        self.screen.blit(speed_text, (10, 10))
        self.screen.blit(gate_text, (10, 50))
        self.screen.blit(reward_text, (10, 90))
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def get_car_corners(self):
        """Get car corner points for drawing"""
        cos_angle = math.cos(math.radians(self.car_angle))
        sin_angle = math.sin(math.radians(self.car_angle))
        
        corners = [
            (-self.car_width // 2, -self.car_height // 2),
            (self.car_width // 2, -self.car_height // 2),
            (self.car_width // 2, self.car_height // 2),
            (-self.car_width // 2, self.car_height // 2)
        ]
        
        rotated_corners = []
        for x, y in corners:
            rotated_x = x * cos_angle - y * sin_angle + self.car_x
            rotated_y = x * sin_angle + y * cos_angle + self.car_y
            rotated_corners.append((rotated_x, rotated_y))
        
        return rotated_corners