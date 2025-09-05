import pygame
import math
import numpy as np

class SimpleDrivingGame:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("AI Driving Game")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.GRAY = (128, 128, 128)
        
        # Car properties
        self.car_x = width // 2
        self.car_y = height - 100
        self.car_angle = 0
        self.car_speed = 0
        self.max_speed = 8
        self.car_width = 20
        self.car_height = 40
        
        # Track properties - now curved!
        self.track_width = 200
        self.track_base_center = width // 2
        self.track_curve_intensity = 150  # How much the track curves
        self.track_curve_frequency = 0.005  # How often it curves
        
        # Game state
        self.running = True
        self.crashed = False
        self.distance_traveled = 0
        self.game_time = 0  # For track generation
        
    def get_track_center_at_y(self, y):
        """Calculate where the track center should be at a given Y position"""
        # Create a sinusoidal curve
        curve_offset = math.sin(y * self.track_curve_frequency) * self.track_curve_intensity
        return self.track_base_center + curve_offset
    
    def get_track_bounds_at_y(self, y):
        """Get left and right track boundaries at given Y position"""
        center = self.get_track_center_at_y(y)
        left = center - self.track_width // 2
        right = center + self.track_width // 2
        return left, right
        
    def reset(self):
        """Reset the game to initial state"""
        self.car_x = self.width // 2
        self.car_y = self.height - 100
        self.car_angle = 0
        self.car_speed = 0
        self.crashed = False
        self.distance_traveled = 0
        self.game_time = 0
        self.previous_y = self.car_y
        return self.get_state()
    
    def get_state(self):
        """Get current game state for AI - now includes upcoming track info"""
        # Get current track boundaries
        current_left, current_right = self.get_track_bounds_at_y(self.car_y)
        
        # Look ahead to see upcoming curve
        look_ahead_distance = 100
        future_left, future_right = self.get_track_bounds_at_y(self.car_y - look_ahead_distance)
        
        # Calculate distances
        left_distance = max(0, self.car_x - current_left) / (self.track_width // 2)
        right_distance = max(0, current_right - self.car_x) / (self.track_width // 2)
        
        # Future track direction (where track is heading)
        current_center = self.get_track_center_at_y(self.car_y)
        future_center = self.get_track_center_at_y(self.car_y - look_ahead_distance)
        track_direction = (future_center - current_center) / self.track_curve_intensity
        
        # Normalize values
        speed = self.car_speed / self.max_speed
        angle = self.car_angle / 360
        
        return np.array([left_distance, right_distance, speed, angle, track_direction])
    
    def step(self, action):
        """Execute one game step based on action"""
        # Actions: 0=nothing, 1=left, 2=right, 3=accelerate, 4=brake
        if action == 1:  # Turn left
            self.car_angle -= 5
        elif action == 2:  # Turn right
            self.car_angle += 5
        elif action == 3:  # Accelerate
            self.car_speed = min(self.car_speed + 1, self.max_speed)
        elif action == 4:  # Brake
            self.car_speed = max(self.car_speed - 1, 0)
        
        # Apply friction
        self.car_speed *= 0.98
        
        # Move car
        angle_rad = math.radians(self.car_angle)
        self.car_x += self.car_speed * math.sin(angle_rad)
        self.car_y -= self.car_speed * math.cos(angle_rad)
        
        # Update distance traveled and game time
        self.distance_traveled += self.car_speed
        self.game_time += 1
        
        # Check for crashes (off curved track)
        track_left, track_right = self.get_track_bounds_at_y(self.car_y)
        
        if self.car_x < track_left or self.car_x > track_right:
            self.crashed = True
        
        # Calculate reward
        reward = self.calculate_reward()
        
        # Check if episode is done
        done = self.crashed or self.car_y < 0
        
        return self.get_state(), reward, done
    
    def calculate_reward(self):
        """Calculate reward for current state"""
        if self.crashed:
            return -100  # Big penalty for crashing
        
        # STRONG reward for forward movement (based on speed)
        speed_reward = self.car_speed * 3.0
        
        # PENALTY for being too slow or stationary
        if self.car_speed < 0.5:
            speed_reward -= 2.0
        
        # Reward for staying on track (more important now with curves)
        track_left, track_right = self.get_track_bounds_at_y(self.car_y)
        track_center = (track_left + track_right) / 2
        track_center_distance = abs(self.car_x - track_center)
        center_bonus = max(0, (self.track_width // 2 - track_center_distance)) * 0.02
        
        # Bonus for forward progress
        if hasattr(self, 'previous_y'):
            if self.car_y < self.previous_y:
                progress_bonus = (self.previous_y - self.car_y) * 0.5
            else:
                progress_bonus = -0.5
        else:
            progress_bonus = 0
        
        self.previous_y = self.car_y
        
        # Bonus for appropriate steering (steering towards track center)
        track_center = self.get_track_center_at_y(self.car_y)
        if self.car_x < track_center and self.car_angle > 0:  # Left of center, steering right
            steering_bonus = 0.1
        elif self.car_x > track_center and self.car_angle < 0:  # Right of center, steering left
            steering_bonus = 0.1
        else:
            steering_bonus = 0
        
        return speed_reward + center_bonus + progress_bonus + steering_bonus
    
    def render(self):
        """Render the game with curved track"""
        self.screen.fill(self.GREEN)  # Grass
        
        # Draw curved track
        track_points_left = []
        track_points_right = []
        
        for y in range(0, self.height, 5):  # Draw track in segments
            left, right = self.get_track_bounds_at_y(y)
            track_points_left.append((left, y))
            track_points_right.append((right, y))
        
        # Draw track surface
        if len(track_points_left) > 2:
            track_points = track_points_left + list(reversed(track_points_right))
            pygame.draw.polygon(self.screen, self.GRAY, track_points)
        
        # Draw center line
        for y in range(0, self.height, 20):
            center_x = self.get_track_center_at_y(y)
            pygame.draw.circle(self.screen, self.WHITE, (int(center_x), y), 3)
        
        # Draw car
        if not self.crashed:
            car_points = self.get_car_corners()
            pygame.draw.polygon(self.screen, self.RED, car_points)
        
        # Draw info
        font = pygame.font.Font(None, 36)
        speed_text = font.render(f"Speed: {self.car_speed:.1f}", True, self.WHITE)
        distance_text = font.render(f"Distance: {int(self.distance_traveled)}", True, self.WHITE)
        self.screen.blit(speed_text, (10, 10))
        self.screen.blit(distance_text, (10, 50))
        
        if self.crashed:
            crash_text = font.render("CRASHED!", True, self.RED)
            self.screen.blit(crash_text, (self.width // 2 - 60, self.height // 2))
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def get_car_corners(self):
        """Get car corner points for drawing"""
        cos_angle = math.cos(math.radians(self.car_angle))
        sin_angle = math.sin(math.radians(self.car_angle))
        
        # Car corners relative to center
        corners = [
            (-self.car_width // 2, -self.car_height // 2),
            (self.car_width // 2, -self.car_height // 2),
            (self.car_width // 2, self.car_height // 2),
            (-self.car_width // 2, self.car_height // 2)
        ]
        
        # Rotate and translate corners
        rotated_corners = []
        for x, y in corners:
            rotated_x = x * cos_angle - y * sin_angle + self.car_x
            rotated_y = x * sin_angle + y * cos_angle + self.car_y
            rotated_corners.append((rotated_x, rotated_y))
        
        return rotated_corners