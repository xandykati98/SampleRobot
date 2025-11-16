import pygame
import math
import numpy as np
from typing import List
from simulation import Robot, World


class Visualizer:
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    ORANGE = (255, 165, 0)
    PURPLE = (128, 0, 128)
    PINK = (255, 192, 203)
    LIME = (0, 255, 0)
    
    ROBOT_COLORS = [
        RED, GREEN, BLUE, YELLOW, CYAN, 
        MAGENTA, ORANGE, PURPLE, PINK, LIME
    ]
    
    def __init__(self, world: World, scale: float = 0.35):
        self.world = world
        self.scale = scale
        self.screen_width = int(world.width * scale)
        self.screen_height = int(world.height * scale)
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height + 100))
        pygame.display.set_caption("Robot Training Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
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
                   draw_sonar: bool = False):
        if not robot.alive:
            # Draw dead robots as transparent
            color = tuple(c // 2 for c in color)
        
        x, y = self.world_to_screen(robot.x, robot.y)
        
        # Check for valid coordinates
        if not (0 <= x <= self.screen_width and 0 <= y <= self.screen_height):
            return  # Robot is off-screen, skip drawing
        
        radius = int(robot.ROBOT_RADIUS * self.scale)
        
        # Draw robot body
        pygame.draw.circle(self.screen, color, (x, y), radius)
        pygame.draw.circle(self.screen, self.BLACK, (x, y), radius, 1)
        
        # Draw direction indicator
        end_x = x + radius * math.cos(math.radians(robot.theta))
        end_y = y + radius * math.sin(math.radians(robot.theta))
        pygame.draw.line(self.screen, self.BLACK, (x, y), (int(end_x), int(end_y)), 2)
        
        # Draw sonar rays (only for alive robots and if enabled)
        if robot.alive and draw_sonar:
            for i, angle in enumerate(robot.SONAR_ANGLES):
                sonar_angle = robot.theta + angle
                sonar_rad = math.radians(sonar_angle)
                distance = robot.sonar_readings[i]
                
                # Clamp distance to reasonable values
                distance = min(distance, robot.MAX_SONAR_RANGE)
                
                end_x = x + distance * self.scale * math.cos(sonar_rad)
                end_y = y + distance * self.scale * math.sin(sonar_rad)
                
                # Validate coordinates before drawing
                if math.isfinite(end_x) and math.isfinite(end_y):
                    end_x = int(max(0, min(self.screen_width, end_x)))
                    end_y = int(max(0, min(self.screen_height, end_y)))
                    
                    # Use different color based on distance
                    if distance < 500:
                        ray_color = self.RED
                    elif distance < 1500:
                        ray_color = self.YELLOW
                    else:
                        ray_color = self.GREEN
                    
                    pygame.draw.line(self.screen, ray_color, (x, y), (end_x, end_y), 1)
    
    def draw_robots(self, robots: List[Robot], draw_sonar_for_best: bool = True):
        # Draw all robots
        for i, robot in enumerate(robots):
            color = self.ROBOT_COLORS[i % len(self.ROBOT_COLORS)]
            # Only draw sonar for the first (best) robot
            draw_sonar = draw_sonar_for_best and i == 0
            self.draw_robot(robot, color, draw_sonar)
    
    def draw_sensor_values(self, robot: Robot):
        # Draw sensor values for the red agent (robot with sensors visualized)
        if not robot.alive:
            return
        
        # Draw sensor values in top-left corner
        x_start = 10
        y_start = 10
        
        # Background for readability
        sensor_bg_width = 200
        sensor_bg_height = 200
        bg_surface = pygame.Surface((sensor_bg_width, sensor_bg_height))
        bg_surface.set_alpha(200)
        bg_surface.fill(self.BLACK)
        self.screen.blit(bg_surface, (x_start - 5, y_start - 5))
        
        # Title
        title_text = self.small_font.render("Red Agent Sensors:", True, self.WHITE)
        self.screen.blit(title_text, (x_start, y_start))
        
        # Draw each sensor value with color matching sensor ray color
        y_offset = y_start + 25
        for i, angle in enumerate(robot.SONAR_ANGLES):
            distance = robot.sonar_readings[i]
            
            # Use same color logic as sensor rays
            if distance < 200:
                text_color = self.RED
            elif distance < 500:
                text_color = self.YELLOW
            else:
                text_color = self.GREEN
            
            sensor_text = self.small_font.render(
                f"S{i} ({angle:+3d}°): {distance:6.0f} mm", 
                True, text_color
            )
            self.screen.blit(sensor_text, (x_start, y_offset))
            y_offset += 20
    
    def draw_stats(self, robots: List[Robot], generation: int, elapsed_time: float):
        # Draw stats panel at the bottom
        panel_y = self.screen_height
        pygame.draw.rect(self.screen, self.GRAY, (0, panel_y, self.screen_width, 100))
        
        # Generation and time
        gen_text = self.font.render(f"Generation: {generation}", True, self.WHITE)
        time_text = self.font.render(f"Time: {elapsed_time:.1f}s", True, self.WHITE)
        self.screen.blit(gen_text, (10, panel_y + 10))
        self.screen.blit(time_text, (10, panel_y + 35))
        
        # Robot stats
        alive_count = sum(1 for r in robots if r.alive)
        alive_text = self.font.render(f"Alive: {alive_count}/{len(robots)}", True, self.WHITE)
        self.screen.blit(alive_text, (200, panel_y + 10))
        
        # Best fitness
        if robots:
            best_fitness = max(r.fitness for r in robots)
            fitness_text = self.font.render(f"Best Fitness: {best_fitness:.1f}", True, self.WHITE)
            self.screen.blit(fitness_text, (200, panel_y + 35))
        
        # Individual robot fitness (top 5)
        x_offset = 400
        for i in range(min(5, len(robots))):
            color = self.ROBOT_COLORS[i % len(self.ROBOT_COLORS)]
            status = "A" if robots[i].alive else "D"
            robot_text = self.small_font.render(
                f"R{i}: {robots[i].fitness:.0f} ({status})", 
                True, color
            )
            y_pos = panel_y + 10 + (i % 5) * 18
            self.screen.blit(robot_text, (x_offset, y_pos))
    
    def update(self, robots: List[Robot], generation: int, elapsed_time: float):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Draw everything
        self.draw_world()
        self.draw_robots(robots, draw_sonar_for_best=True)
        self.draw_stats(robots, generation, elapsed_time)
        
        # Draw sensor values for red agent (first robot) if alive
        if robots and robots[0].alive:
            self.draw_sensor_values(robots[0])
        
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS
        
        return True
    
    def close(self):
        pygame.quit()

