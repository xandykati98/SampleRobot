import numpy as np
import pygame
import math
import json
import time
from typing import List, Tuple, Dict
from simulation import Robot, World


class DataExtractor:
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    
    # Direction actions: (left_vel, right_vel, name)
    # Using arc movements instead of pure rotation for better training data
    DIRECTIONS = [
        (300, 300, "forward"),          # Straight forward
        (250, 300, "forward_left"),     # Gentle arc left while moving forward
        (300, 250, "forward_right"),    # Gentle arc right while moving forward
    ]
    
    def __init__(self, world: World, scale: float = 0.35):
        self.world = world
        self.scale = scale
        self.screen_width = int(world.width * scale)
        self.screen_height = int(world.height * scale)
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height + 150))
        pygame.display.set_caption("Data Extraction - Testing Directions")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        self.dataset = []
        self.test_duration = 1.75  # 3 seconds per direction
        self.dt = 0.05
        
    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        screen_x = int(x * self.scale)
        screen_y = int(y * self.scale)
        return screen_x, screen_y
    
    def draw_world(self):
        self.screen.fill(self.WHITE)
        
        # Draw obstacles
        for obstacle in self.world.obstacles:
            x, y = self.world_to_screen(obstacle.x, obstacle.y)
            w = int(obstacle.width * self.scale)
            h = int(obstacle.height * self.scale)
            pygame.draw.rect(self.screen, self.GRAY, (x, y, w, h))
            pygame.draw.rect(self.screen, self.BLACK, (x, y, w, h), 1)
    
    def draw_robot(self, robot: Robot, color: tuple[int, int, int], 
                   draw_trail: bool = False, trail_points: List[Tuple[float, float]] = None):
        x, y = self.world_to_screen(robot.x, robot.y)
        
        if not (0 <= x <= self.screen_width and 0 <= y <= self.screen_height):
            return
        
        radius = int(robot.ROBOT_RADIUS * self.scale)
        
        # Draw trail if requested
        if draw_trail and trail_points:
            if len(trail_points) > 1:
                screen_points = [self.world_to_screen(px, py) for px, py in trail_points]
                pygame.draw.lines(self.screen, color, False, screen_points, 2)
        
        # Draw robot body
        pygame.draw.circle(self.screen, color, (x, y), radius)
        pygame.draw.circle(self.screen, self.BLACK, (x, y), radius, 1)
        
        # Draw direction indicator
        end_x = x + radius * math.cos(math.radians(robot.theta))
        end_y = y + radius * math.sin(math.radians(robot.theta))
        pygame.draw.line(self.screen, self.BLACK, (x, y), (int(end_x), int(end_y)), 2)
    
    def test_direction(self, start_x: float, start_y: float, start_theta: float,
                      left_vel: float, right_vel: float, direction_name: str) -> Dict:
        """Test a direction for test_duration seconds and return results"""
        robot = Robot(start_x, start_y, start_theta)
        robot.update_sonar(self.world)
        
        initial_distance = robot.distance_traveled
        trail_points = [(robot.x, robot.y)]
        sim_time = 0.0
        steps = 0
        
        while sim_time < self.test_duration:
            if not robot.alive:
                break
            
            # Execute action
            robot.update(left_vel, right_vel, self.dt, self.world)
            robot.update_sonar(self.world)
            
            # Record trail point every 0.2 seconds
            if len(trail_points) == 0 or math.sqrt(
                (robot.x - trail_points[-1][0])**2 + (robot.y - trail_points[-1][1])**2
            ) > 20:
                trail_points.append((robot.x, robot.y))
            
            sim_time += self.dt
            steps += 1
        
        final_distance = robot.distance_traveled
        distance_traveled = final_distance - initial_distance
        
        return {
            'alive': robot.alive,
            'distance_traveled': distance_traveled,
            'trail_points': trail_points,
            'final_x': robot.x,
            'final_y': robot.y,
            'direction_name': direction_name,
            'steps': steps
        }
    
    def evaluate_best_direction(self, start_x: float, start_y: float, start_theta: float) -> Tuple[int, List[Dict]]:
        """Test all 4 directions and return the best one"""
        results = []
        
        for left_vel, right_vel, direction_name in self.DIRECTIONS:
            result = self.test_direction(start_x, start_y, start_theta, left_vel, right_vel, direction_name)
            results.append(result)
        
        # Find best direction: prioritize forward if alive, otherwise best distance
        # Index 0 = forward, 1 = forward_left, 2 = forward_right
        
        # If forward is alive, always prefer it over sides
        if results[0]['alive']:
            best_idx = 0
        else:
            # Forward is dead, pick best among sides based on distance
            best_idx = 1  # Default to forward_left
            best_score = -float('inf')
            
            for i in [1, 2]:  # Check forward_left and forward_right
                if results[i]['alive']:
                    score = results[i]['distance_traveled']
                    if score > best_score:
                        best_score = score
                        best_idx = i
                else:
                    score = -1000  # Dead robots get very low score
                    if score > best_score:
                        best_score = score
                        best_idx = i
        
        return best_idx, results
    
    def visualize_test(self, start_x: float, start_y: float, start_theta: float,
                      results: List[Dict], best_idx: int, quick_mode: bool = True):
        """Visualize all 3 direction tests"""
        colors = [self.RED, self.BLUE, self.YELLOW]
        
        running = True
        frame = 0
        max_frames = 10 if quick_mode else 180  # Show for 0.16s or 6s
        
        while running and frame < max_frames:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    # Skip to next sample with spacebar
                    return True
            
            # Draw world
            self.draw_world()
            
            # Draw all trails
            for i, result in enumerate(results):
                color = colors[i]
                if i == best_idx:
                    # Make best direction brighter
                    color = tuple(min(255, c + 50) for c in color)
                
                if result['trail_points']:
                    screen_points = [self.world_to_screen(px, py) for px, py in result['trail_points']]
                    if len(screen_points) > 1:
                        pygame.draw.lines(self.screen, color, False, screen_points, 3)
            
            # Draw starting position (large, black)
            start_robot = Robot(start_x, start_y, start_theta)
            x, y = self.world_to_screen(start_robot.x, start_robot.y)
            if 0 <= x <= self.screen_width and 0 <= y <= self.screen_height:
                radius = int(start_robot.ROBOT_RADIUS * self.scale)
                pygame.draw.circle(self.screen, self.BLACK, (x, y), radius)
                pygame.draw.circle(self.screen, self.WHITE, (x, y), radius, 2)
            
            # Draw robots at final positions (small circles with labels)
            for i, result in enumerate(results):
                color = colors[i]
                if i == best_idx:
                    color = tuple(min(255, c + 50) for c in color)
                
                # Draw final position as a small circle
                fx, fy = self.world_to_screen(result['final_x'], result['final_y'])
                if 0 <= fx <= self.screen_width and 0 <= fy <= self.screen_height:
                    pygame.draw.circle(self.screen, color, (fx, fy), 8, 0)
                    pygame.draw.circle(self.screen, self.BLACK, (fx, fy), 8, 1)
                    
                    # Add label for each direction
                    direction_labels = {0: "F", 1: "FL", 2: "FR"}
                    label = self.small_font.render(direction_labels.get(i, str(i)), True, self.BLACK)
                    self.screen.blit(label, (fx - 5, fy - 5))
            
            # Draw info panel
            panel_y = self.screen_height
            pygame.draw.rect(self.screen, self.GRAY, (0, panel_y, self.screen_width, 150))
            
            # Best direction info
            best_result = results[best_idx]
            best_text = self.font.render(
                f"Best: {best_result['direction_name']} | "
                f"Distance: {best_result['distance_traveled']:.1f}mm | "
                f"Alive: {best_result['alive']}",
                True, self.WHITE
            )
            self.screen.blit(best_text, (10, panel_y + 10))
            
            # All directions info
            y_offset = panel_y + 40
            for i, result in enumerate(results):
                color = colors[i]
                status = "✓" if result['alive'] else "✗"
                dir_text = self.small_font.render(
                    f"{result['direction_name']}: {result['distance_traveled']:.1f}mm {status}",
                    True, color
                )
                self.screen.blit(dir_text, (10, y_offset + i * 25))
            
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS instead of 30
            
            frame += 1
        
        return True
    
    def collect_data(self, num_samples: int = 100, visualize: bool = True, quick_vis: bool = True):
        """Collect data samples from random positions"""
        print(f"Collecting {num_samples} data samples...")
        if visualize:
            if quick_vis:
                print("Quick visualization mode (press SPACE to skip, close window to stop)")
            else:
                print("Full visualization mode (close window to stop)")
        else:
            print("No visualization mode (fast collection)")
        
        for sample_idx in range(num_samples):
            # Generate random starting position (avoid obstacles)
            max_attempts = 50
            start_x, start_y = 0, 0
            valid_position = False
            
            for _ in range(max_attempts):
                start_x = np.random.uniform(100, self.world.width - 100)
                start_y = np.random.uniform(100, self.world.height - 100)
                start_theta = np.random.uniform(0, 360)
                
                # Check if position is valid (not inside obstacle)
                test_robot = Robot(start_x, start_y, start_theta)
                if not test_robot.check_collision(self.world):
                    valid_position = True
                    break
            
            if not valid_position:
                print(f"  Sample {sample_idx}: Could not find valid position, skipping")
                continue
            
            # Get initial sensor readings
            test_robot.update_sonar(self.world)
            sensor_readings = test_robot.get_normalized_sonar().tolist()
            
            # Test all directions
            best_idx, results = self.evaluate_best_direction(start_x, start_y, start_theta)
            
            # Store data sample
            sample = {
                'sensor_readings': sensor_readings,
                'best_direction': best_idx,
                'best_direction_name': self.DIRECTIONS[best_idx][2],
                'start_x': start_x,
                'start_y': start_y,
                'start_theta': start_theta,
            'results': {
                'forward': results[0],
                'forward_left': results[1],
                'forward_right': results[2]
            }
            }
            
            self.dataset.append(sample)
            
            # Visualize (if enabled)
            if visualize:
                if not self.visualize_test(start_x, start_y, start_theta, results, best_idx, quick_mode=quick_vis):
                    print("Data collection interrupted by user")
                    break
            
            # Print progress
            if (sample_idx + 1) % 10 == 0 or not visualize:
                print(f"  Collected {sample_idx + 1}/{num_samples} samples")
        
        print(f"\nData collection complete! Collected {len(self.dataset)} samples")
    
    def save_dataset(self, filename: str = 'training_data.json'):
        """Save dataset to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.dataset, f, indent=2)
        print(f"Dataset saved to {filename}")
        print(f"Total samples: {len(self.dataset)}")
        
        # Print statistics
        direction_counts = {}
        for sample in self.dataset:
            dir_name = sample['best_direction_name']
            direction_counts[dir_name] = direction_counts.get(dir_name, 0) + 1
        
        print("\nBest direction distribution:")
        for direction, count in direction_counts.items():
            print(f"  {direction}: {count} ({count/len(self.dataset)*100:.1f}%)")
    
    def close(self):
        pygame.quit()


def main():
    # Create world with more obstacles for harder navigation
    world = World(width=2000, height=2000, num_obstacles=30)
    
    # Create data extractor
    extractor = DataExtractor(world, scale=0.35)
    
    # Collect data
    # Options:
    # visualize=True, quick_vis=True: Quick visualization (0.16s per sample) - RECOMMENDED
    # visualize=True, quick_vis=False: Full visualization (6s per sample)
    # visualize=False: No visualization (fastest, ~instant per sample)
    extractor.collect_data(num_samples=2000, visualize=False, quick_vis=False)
    
    # Save dataset
    extractor.save_dataset('training_data.json')
    
    extractor.close()


if __name__ == "__main__":
    main()

