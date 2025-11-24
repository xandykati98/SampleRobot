import numpy as np
import math
from typing import List, Tuple


class Obstacle:
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
    def contains_point(self, px: float, py: float) -> bool:
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + self.height)
    
    def intersects_line(self, x1: float, y1: float, x2: float, y2: float) -> Tuple[bool, float]:
        # Line-rectangle intersection using Liang-Barsky algorithm
        # Returns (intersects, distance)
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return False, float('inf')
        
        t_min = 0.0
        t_max = 1.0
        
        # Left, right, bottom, top edges
        edges = [
            (-dx, x1 - self.x),
            (dx, self.x + self.width - x1),
            (-dy, y1 - self.y),
            (dy, self.y + self.height - y1)
        ]
        
        for p, q in edges:
            if p == 0:
                if q < 0:
                    return False, float('inf')
            else:
                t = q / p
                if p < 0:
                    t_min = max(t_min, t)
                else:
                    t_max = min(t_max, t)
                    
                if t_min > t_max:
                    return False, float('inf')
        
        if t_min <= t_max:
            distance = math.sqrt(dx**2 + dy**2) * t_min
            return True, distance
        
        return False, float('inf')


class Robot:
    # Sonar angles matching C++ layout: [90°, 50°, 30°, 10°, -10°, -30°, -50°, -90°]
    SONAR_ANGLES = [90, 50, 30, 10, -10, -30, -50, -90]
    MAX_SONAR_RANGE = 500.0  # Adjusted for simulator scale (C++ uses 5000mm in real world)
    ROBOT_RADIUS = 20.0
    
    def __init__(self, x: float, y: float, theta: float, max_velocity: float = 400.0):
        self.x = x
        self.y = y
        self.theta = theta  # in degrees
        self.max_velocity = max_velocity  # Maximum velocity in mm/s
        self.start_x = x  # Store start position for distance calculation
        self.start_y = y
        self.left_vel = 0.0
        self.right_vel = 0.0
        self.sonar_readings = [self.MAX_SONAR_RANGE] * 8
        self.alive = True
        self.fitness = 0.0
        self.distance_traveled = 0.0
        self.visited_positions = set()
        self.grid_size = 100  # Grid cell size for position tracking
        
        self.last_action = 0  # Track last action (0=forward, 1=left, 2=right)
        
    def update(self, action, dt: float, world):
        if not self.alive:
            return
        
        # Handle both discrete (int) and continuous (tuple/array) actions
        if isinstance(action, (tuple, list, np.ndarray)):
            # Continuous action: (left_vel, right_vel)
            left_vel, right_vel = action
            self.last_action = 0  # Not used for continuous actions
        else:
            # Discrete action: 0=forward, 1=correct_left, 2=correct_right
            self.last_action = action
            
            FORWARD_VEL = 150.0  # Base forward velocity (mm/s)
            TURN_VEL_DIFF = 50.0  # Velocity difference for turning
            
            if action == 0:  # Forward
                left_vel = FORWARD_VEL
                right_vel = FORWARD_VEL
            elif action == 1:  # Correct left
                left_vel = FORWARD_VEL - TURN_VEL_DIFF
                right_vel = FORWARD_VEL + TURN_VEL_DIFF
            elif action == 2:  # Correct right
                left_vel = FORWARD_VEL + TURN_VEL_DIFF
                right_vel = FORWARD_VEL - TURN_VEL_DIFF
            else:
                left_vel = FORWARD_VEL
                right_vel = FORWARD_VEL
        
        # Clamp velocities to prevent teleportation
        left_vel = np.clip(left_vel, 0.0, self.max_velocity)
        right_vel = np.clip(right_vel, 0.0, self.max_velocity)
        
        self.left_vel = left_vel
        self.right_vel = right_vel
        
        # Differential drive kinematics
        wheel_base = 40.0  # Distance between wheels
        
        # Calculate linear and angular velocities
        v = (left_vel + right_vel) / 2.0
        omega = (right_vel - left_vel) / wheel_base
        
        # Additional safety check: limit maximum displacement per timestep
        max_displacement = 50.0  # Maximum 50mm per timestep to prevent teleporting
        if abs(v * dt) > max_displacement:
            v = np.sign(v) * (max_displacement / dt)
        
        old_x = self.x
        old_y = self.y
        
        # Update position
        if abs(omega) < 0.001:
            # Moving straight
            self.x += v * dt * math.cos(math.radians(self.theta))
            self.y += v * dt * math.sin(math.radians(self.theta))
        else:
            # Curved motion
            R = v / omega
            theta_rad = math.radians(self.theta)
            delta_theta = omega * dt  # This is in radians
            
            self.x += R * (math.sin(theta_rad + delta_theta) - math.sin(theta_rad))
            self.y += R * (-math.cos(theta_rad + delta_theta) + math.cos(theta_rad))
            self.theta += math.degrees(delta_theta)
        
        # Normalize theta
        self.theta = self.theta % 360
        
        # Calculate distance traveled
        dist = math.sqrt((self.x - old_x)**2 + (self.y - old_y)**2)
        self.distance_traveled += dist
        
        # Track visited positions for penalty
        grid_x = int(self.x // self.grid_size)
        grid_y = int(self.y // self.grid_size)
        self.visited_positions.add((grid_x, grid_y))
        
        # Check collision
        if self.check_collision(world):
            self.alive = False
            
    def check_collision(self, world) -> bool:
        # Check world boundaries
        if (self.x - self.ROBOT_RADIUS < 0 or 
            self.x + self.ROBOT_RADIUS > world.width or
            self.y - self.ROBOT_RADIUS < 0 or 
            self.y + self.ROBOT_RADIUS > world.height):
            return True
        
        # Check obstacles
        for obstacle in world.obstacles:
            if self.intersects_obstacle(obstacle):
                return True
        
        return False
    
    def intersects_obstacle(self, obstacle: Obstacle) -> bool:
        # Simple circle-rectangle collision
        closest_x = max(obstacle.x, min(self.x, obstacle.x + obstacle.width))
        closest_y = max(obstacle.y, min(self.y, obstacle.y + obstacle.height))
        
        distance = math.sqrt((self.x - closest_x)**2 + (self.y - closest_y)**2)
        
        return distance < self.ROBOT_RADIUS
    
    def update_sonar(self, world):
        if not self.alive:
            self.sonar_readings = [0.0] * 8
            return
        
        for i, angle in enumerate(self.SONAR_ANGLES):
            sonar_angle = self.theta + angle
            sonar_rad = math.radians(sonar_angle)
            
            # Ray endpoint
            end_x = self.x + self.MAX_SONAR_RANGE * math.cos(sonar_rad)
            end_y = self.y + self.MAX_SONAR_RANGE * math.sin(sonar_rad)
            
            min_distance = self.MAX_SONAR_RANGE
            
            # Check world boundaries
            boundary_dist = self.check_boundary_distance(sonar_rad, world)
            if math.isfinite(boundary_dist) and boundary_dist > 0:
                min_distance = min(min_distance, boundary_dist)
            
            # Check obstacles
            for obstacle in world.obstacles:
                intersects, distance = obstacle.intersects_line(
                    self.x, self.y, end_x, end_y
                )
                if intersects and math.isfinite(distance) and distance > 0:
                    min_distance = min(min_distance, distance)
            
            # Ensure valid reading
            self.sonar_readings[i] = max(0.0, min(min_distance, self.MAX_SONAR_RANGE))
    
    def check_boundary_distance(self, sonar_rad: float, world) -> float:
        cos_angle = math.cos(sonar_rad)
        sin_angle = math.sin(sonar_rad)
        
        distances = []
        
        # Left boundary (x = 0)
        if cos_angle < -0.0001:
            dist = -self.x / cos_angle
            if dist > 0:
                distances.append(dist)
        
        # Right boundary (x = world.width)
        if cos_angle > 0.0001:
            dist = (world.width - self.x) / cos_angle
            if dist > 0:
                distances.append(dist)
        
        # Top boundary (y = 0)
        if sin_angle < -0.0001:
            dist = -self.y / sin_angle
            if dist > 0:
                distances.append(dist)
        
        # Bottom boundary (y = world.height)
        if sin_angle > 0.0001:
            dist = (world.height - self.y) / sin_angle
            if dist > 0:
                distances.append(dist)
        
        if distances:
            return min(distances)
        return self.MAX_SONAR_RANGE
    
    def get_normalized_sonar(self) -> np.ndarray:
        # Normalize sonar readings to [0, 1]
        return np.array(self.sonar_readings) / self.MAX_SONAR_RANGE
    
    def calculate_fitness(self) -> float:
        # Calculate distance from start position
        distance_from_start = math.sqrt(
            (self.x - self.start_x)**2 + (self.y - self.start_y)**2
        )
        
        if not self.alive:
            # Penalize dead robots - use 50% of distance from start
            base_fitness = distance_from_start * 0.5
        else:
            # Reward distance from start position
            base_fitness = distance_from_start
        
        # Small bonus for exploration (visited positions)
        exploration_bonus = len(self.visited_positions) * 2.0
        
        self.fitness = base_fitness + exploration_bonus
        return self.fitness
    
    def calculate_reward(self, dt: float) -> float:
        # Penalty for collision
        if not self.alive:
            return -20.0  # Further reduced from -50
        
        # Reward for moving forward (significantly increased)
        forward_velocity = (self.left_vel + self.right_vel) / 2.0
        forward_speed = abs(forward_velocity)
        forward_reward = forward_speed * dt * 0.5  # Increased from 0.1 to 0.5
        
        # Penalty for excessive spinning (only penalize severe spinning)
        angular_velocity = abs(self.right_vel - self.left_vel) / 40.0
        spinning_penalty = 0.0
        
        # Only penalize if spinning very fast AND not moving forward
        if forward_speed < 30.0 and angular_velocity > 1.5:
            # Light penalty only for severe spinning in place
            spinning_penalty = -0.1 * angular_velocity  # Much reduced
        
        # Reward for being in new positions
        grid_x = int(self.x // self.grid_size)
        grid_y = int(self.y // self.grid_size)
        is_new_position = (grid_x, grid_y) not in self.visited_positions
        exploration_reward = 3.0 if is_new_position else 0.0
        
        # Small penalty for obstacles too close (very light)
        min_sonar = min(self.sonar_readings)
        danger_penalty = 0.0
        if min_sonar < 300:  # Only penalize when very close
            danger_penalty = -0.1
        
        # Bonus for forward movement (encourage moving forward)
        forward_bonus = 0.0
        if forward_speed > 50.0:  # Lower threshold
            if angular_velocity < 0.5:
                forward_bonus = 2.0  # Increased bonus for straight movement
            else:
                forward_bonus = 0.5  # Smaller bonus even when turning
        
        # Survival bonus (small positive reward for staying alive)
        survival_bonus = 0.1
        
        total_reward = forward_reward + exploration_reward + danger_penalty + spinning_penalty + forward_bonus + survival_bonus
        
        return total_reward


class World:
    def __init__(self, width: float, height: float, num_obstacles: int = 10):
        self.width = width
        self.height = height
        self.obstacles = []
        self.generate_obstacles(num_obstacles)
        
    def generate_obstacles(self, num_obstacles: int):
        np.random.seed()
        self.obstacles = []
        
        for _ in range(num_obstacles):
            # Random obstacle size and position
            w = np.random.uniform(50, 200)
            h = np.random.uniform(50, 200)
            x = np.random.uniform(0, self.width - w)
            y = np.random.uniform(0, self.height - h)
            
            # Avoid placing obstacles at the center start position
            if not (800 < x < 1200 and 800 < y < 1200):
                self.obstacles.append(Obstacle(x, y, w, h))

