import torch
import numpy as np
import time
from typing import List

import mlflow
import mlflow.pytorch

from model import RobotNet
from simulation import Robot, World
from evolution import EvolutionaryAlgorithm
from visualization import Visualizer


class Trainer:
    def __init__(self, num_generations: int = 100, episode_duration: float = 30.0, 
                 max_velocity: float = 300.0, visualize: bool = True, 
                 vis_update_freq: int = 5, num_obstacles: int = 350):
        self.num_generations = num_generations
        self.episode_duration = episode_duration  # seconds
        self.dt = 0.05  # 50ms timestep
        self.num_obstacles = num_obstacles
        self.max_velocity = max_velocity
        self.visualize = visualize
        self.vis_update_freq = vis_update_freq  # Update visualization every N frames
        
        # Create world
        self.world = World(width=10000, height=10000, num_obstacles=num_obstacles)
        
        # Create visualizer (only if visualization is enabled)
        self.visualizer = Visualizer(self.world, scale=0.35) if visualize else None
        
        # Create evolutionary algorithm
        self.evolution = EvolutionaryAlgorithm(population_size=12, max_velocity=max_velocity)
        
        self.best_fitness_history = []
        self.frame_count = 0
        
    def run_episode(self, models: List[RobotNet], generation: int) -> List[float]:
        # Create robots for each model
        robots = []
        for _ in range(len(models)):
            # Start position at center
            robot = Robot(x=1000, y=1000, theta=np.random.uniform(0, 360), max_velocity=self.max_velocity)
            robots.append(robot)
        
        # Run simulation
        start_time = time.time()
        sim_time = 0.0
        self.frame_count = 0
        
        while sim_time < self.episode_duration:
            # Update each robot
            for i, robot in enumerate(robots):
                if robot.alive:
                    # Update sonar
                    robot.update_sonar(self.world)
                    
                    # Get action from neural network
                    sonar_normalized = robot.get_normalized_sonar()
                    action = models[i].get_action(sonar_normalized, robot.last_action)
                    
                    # Update robot
                    robot.update(action, self.dt, self.world)
                    
                    # Update fitness
                    robot.calculate_fitness()
            
            # Update visualization only periodically (if enabled)
            if self.visualize and self.frame_count % self.vis_update_freq == 0:
                elapsed_real_time = time.time() - start_time
                running = self.visualizer.update(robots, generation, sim_time)
                
                if not running:
                    raise KeyboardInterrupt("User closed window")
            
            # Early stopping: if only one robot is alive and it's not in top 3, stop episode
            alive_robots = [r for r in robots if r.alive]
            if len(alive_robots) == 1:
                # Calculate current fitness scores
                current_fitness = [r.calculate_fitness() for r in robots]
                # Get sorted indices (descending order)
                sorted_indices = np.argsort(current_fitness)[::-1]
                # Get top 3 indices
                top_3_indices = sorted_indices[:3]
                # Check if the single alive robot is in top 3
                alive_robot_idx = next(i for i, r in enumerate(robots) if r.alive)
                if alive_robot_idx not in top_3_indices:
                    rank = np.where(sorted_indices == alive_robot_idx)[0][0] + 1
                    print(f"  Early stop at {sim_time:.1f}s: Only one robot alive (rank {rank}), not in top 3")
                    break
            
            # Check if all robots are dead
            if all(not r.alive for r in robots):
                print(f"  All robots died at {sim_time:.1f}s")
                break
            
            sim_time += self.dt
            self.frame_count += 1
        
        # Calculate final fitness
        fitness_scores = [robot.calculate_fitness() for robot in robots]
        
        return fitness_scores
    
    def train(self):
        print("Starting training...")
        print(f"Generations: {self.num_generations}")
        print(f"Episode duration: {self.episode_duration}s")
        print(f"Population size: {self.evolution.population_size}")
        print(f"Visualization: {'ON' if self.visualize else 'OFF'}")
        if self.visualize:
            print(f"Visualization update frequency: every {self.vis_update_freq} frames")
        print()
        
        # Start MLflow run
        mlflow.set_experiment("robot_training")
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                "num_generations": self.num_generations,
                "episode_duration": self.episode_duration,
                "population_size": self.evolution.population_size,
                "max_velocity": self.max_velocity,
                "num_obstacles": self.num_obstacles,
                "dt": self.dt,
                "vis_update_freq": self.vis_update_freq
            })
            
            try:
                for generation in range(self.num_generations):
                    # Regenerate obstacles each generation for variety
                    if generation > 0 and generation % 10 == 0:
                        self.world.generate_obstacles(self.num_obstacles)
                    
                    # Run episode with current population
                    episode_start = time.time()
                    fitness_scores = self.run_episode(
                        self.evolution.population, 
                        generation
                    )
                    episode_time = time.time() - episode_start
                    
                    # Track best fitness
                    best_fitness = max(fitness_scores)
                    avg_fitness = sum(fitness_scores) / len(fitness_scores)
                    max_distance = best_fitness  # Fitness is distance from start
                    self.best_fitness_history.append(best_fitness)
                    
                    # Log metrics to MLflow
                    mlflow.log_metrics({
                        "best_fitness": best_fitness,
                        "avg_fitness": avg_fitness,
                        "max_distance": max_distance,
                        "episode_time": episode_time
                    }, step=generation)
                    
                    # Evolve population
                    self.evolution.evolve(fitness_scores)
                    
                    # Print progress (more detailed when visualization is off)
                    if not self.visualize:
                        print(f"  Episode time: {episode_time:.2f}s | "
                              f"Avg fitness: {avg_fitness:.1f} | "
                              f"Best: {best_fitness:.1f}")
                    print()
            
            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
            
            finally:
                # Log final best metrics
                if self.best_fitness_history:
                    final_best_fitness = max(self.best_fitness_history)
                    final_max_distance = final_best_fitness
                    mlflow.log_metrics({
                        "final_best_fitness": final_best_fitness,
                        "final_max_distance": final_max_distance
                    })
                
                # Save best model
                best_model = self.evolution.get_best_model()
                torch.save(best_model.state_dict(), 'best_robot.pth')
                
                # Log model artifact
                mlflow.pytorch.log_model(best_model, "model")
                
                print(f"\nBest model saved to 'best_robot.pth'")
                if self.best_fitness_history:
                    print(f"Best fitness achieved: {max(self.best_fitness_history):.2f}")
                
                if self.visualizer:
                    self.visualizer.close()


def main():
    # Using max_velocity=300 during training for stability (prevents teleporting)
    # Can be increased to 400 for final training or C++ deployment
    
    # Training options:
    # visualize=False: Fast training without visualization (recommended for long training)
    # visualize=True, vis_update_freq=10: Show visualization but update less frequently
    # visualize=True, vis_update_freq=1: Full visualization (slower, for debugging)
    
    trainer = Trainer(
        num_generations=20, 
        episode_duration=90.0, 
        max_velocity=300.0,
        visualize=False,  # Set to True to enable visualization
        vis_update_freq=5  # Update visualization every N frames (only if visualize=True)
    )
    trainer.train()


if __name__ == "__main__":
    main()

