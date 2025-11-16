import torch
import numpy as np
import time
from typing import List

from rl_trainer import ActorCriticNet, PPOTrainer
from simulation import Robot, World
from visualization import Visualizer


class RLTrainer:
    def __init__(self, num_episodes: int = 1000, episode_duration: float = 30.0, 
                 max_velocity: float = 300.0, num_parallel_robots: int = 5,
                 update_frequency: int = 2048):
        self.num_episodes = num_episodes
        self.episode_duration = episode_duration
        self.dt = 0.05  # 50ms timestep
        self.max_velocity = max_velocity
        self.num_parallel_robots = num_parallel_robots
        self.update_frequency = update_frequency
        
        # Create model and trainer
        self.model = ActorCriticNet(hidden_size=32, max_velocity=max_velocity)
        self.ppo_trainer = PPOTrainer(
            self.model, 
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            epochs=10,
            batch_size=64
        )
        
        # Create world
        self.world = World(width=2000, height=2000, num_obstacles=15)
        
        # Create visualizer
        self.visualizer = Visualizer(self.world, scale=0.35)
        
        self.episode_rewards = []
        self.timesteps = 0
        
    def run_episode(self, episode: int) -> float:
        # Create multiple robots for parallel experience collection
        robots = []
        for _ in range(self.num_parallel_robots):
            robot = Robot(x=1000, y=1000, theta=np.random.uniform(0, 360), max_velocity=self.max_velocity)
            robots.append(robot)
        
        episode_reward = 0
        sim_time = 0.0
        start_time = time.time()
        
        # Initialize sonar readings for all robots
        for robot in robots:
            robot.update_sonar(self.world)
        
        while sim_time < self.episode_duration:
            all_done = True
            
            # Update each robot
            for robot in robots:
                if robot.alive:
                    all_done = False
                    
                    # Get current state
                    state = robot.get_normalized_sonar()
                    
                    # Get action from policy
                    action, log_prob, value = self.model.get_action(state, deterministic=False)
                    left_vel, right_vel = action
                    
                    # Execute action
                    robot.update(left_vel, right_vel, self.dt, self.world)
                    robot.update_sonar(self.world)
                    
                    # Calculate reward
                    reward = robot.calculate_reward(self.dt)
                    episode_reward += reward
                    
                    # Store transition in buffer
                    next_state = robot.get_normalized_sonar()
                    done = not robot.alive
                    
                    self.ppo_trainer.buffer.add(state, action, reward, value, log_prob, done)
                    self.timesteps += 1
                    
                    # Update policy when buffer is full
                    if self.timesteps % self.update_frequency == 0:
                        # Get final value for bootstrapping
                        if robot.alive:
                            _, _, next_value = self.model.get_action(next_state, deterministic=True)
                        else:
                            next_value = 0.0
                        
                        stats = self.ppo_trainer.update(next_value)
                        print(f"  Update at timestep {self.timesteps}: " +
                              f"Policy Loss: {stats['policy_loss']:.3f}, " +
                              f"Value Loss: {stats['value_loss']:.3f}, " +
                              f"Entropy: {stats['entropy']:.3f}")
            
            # Update visualization (show only first few robots)
            running = self.visualizer.update(robots[:min(10, len(robots))], episode, sim_time)
            
            if not running:
                raise KeyboardInterrupt("User closed window")
            
            # Check if all robots are dead
            if all_done:
                print(f"  All robots died at {sim_time:.1f}s")
                break
            
            sim_time += self.dt
        
        return episode_reward / self.num_parallel_robots
    
    def train(self):
        print("Starting RL Training with PPO...")
        print(f"Episodes: {self.num_episodes}")
        print(f"Episode duration: {self.episode_duration}s")
        print(f"Parallel robots: {self.num_parallel_robots}")
        print(f"Update frequency: {self.update_frequency} timesteps")
        print()
        
        try:
            for episode in range(self.num_episodes):
                # Regenerate obstacles periodically
                if episode > 0 and episode % 10 == 0:
                    self.world.generate_obstacles(15)
                
                # Run episode
                avg_reward = self.run_episode(episode)
                self.episode_rewards.append(avg_reward)
                
                # Print progress
                recent_avg = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else avg_reward
                print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, " +
                      f"Recent 10 Avg: {recent_avg:.2f}, Timesteps: {self.timesteps}")
                
                # Save checkpoint every 50 episodes
                if (episode + 1) % 50 == 0:
                    torch.save(self.model.state_dict(), f'checkpoint_episode_{episode+1}.pth')
                    print(f"Checkpoint saved at episode {episode+1}")
                
                print()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            # Save final model
            torch.save(self.model.state_dict(), 'best_robot_rl.pth')
            print(f"\nFinal model saved to 'best_robot_rl.pth'")
            print(f"Total timesteps: {self.timesteps}")
            print(f"Average reward: {np.mean(self.episode_rewards):.2f}")
            
            self.visualizer.close()


def main():
    trainer = RLTrainer(
        num_episodes=1000, 
        episode_duration=30.0, 
        max_velocity=300.0,
        num_parallel_robots=5,
        update_frequency=2048
    )
    trainer.train()


if __name__ == "__main__":
    main()

