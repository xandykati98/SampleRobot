import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import List
import random
from collections import deque

from simulation import Robot, World
from visualization import Visualizer


class SimpleQNetwork(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, output_size=3):
        super(SimpleQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, state_size=8, action_size=3, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        
        # Neural networks
        self.q_network = SimpleQNetwork(state_size, 64, action_size)
        self.target_network = SimpleQNetwork(state_size, 64, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copy weights to target network
        self.update_target_network()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class NeuralRLTrainer:
    def __init__(self, num_episodes: int = 200, episode_duration: float = 30.0, 
                 max_velocity: float = 300.0):
        self.num_episodes = num_episodes
        self.episode_duration = episode_duration
        self.dt = 0.1  # 100ms timestep
        self.max_velocity = max_velocity
        
        # Create world
        self.world = World(width=4000, height=4000, num_obstacles=20)
        
        # Create visualizer
        self.visualizer = Visualizer(self.world, scale=0.35)
        
        # Create DQN agent
        self.agent = DQNAgent()
        
        self.episode_rewards = []
        self.episode_lengths = []
        
    def run_episode(self, episode: int) -> tuple:
        # Create robot
        robot = Robot(x=2000, y=2000, theta=np.random.uniform(0, 360), 
                     max_velocity=self.max_velocity)
        
        episode_reward = 0
        sim_time = 0.0
        steps = 0
        
        # Initialize sonar readings
        robot.update_sonar(self.world)
        state = robot.get_normalized_sonar()
        
        while sim_time < self.episode_duration and robot.alive:
            # Get action from agent
            action = self.agent.act(state)
            
            # Execute action
            robot.update(action, self.dt, self.world)
            robot.update_sonar(self.world)
            
            # Get next state and reward
            next_state = robot.get_normalized_sonar()
            reward = robot.calculate_reward(self.dt)
            
            # Modify reward for learning
            if not robot.alive:
                reward = -50.0  # Big penalty for collision
            
            episode_reward += reward
            done = not robot.alive
            
            # Store experience
            self.agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            steps += 1
            
            # Update visualization (show single robot in list)
            running = self.visualizer.update([robot], episode, sim_time)
            
            if not running:
                raise KeyboardInterrupt("User closed window")
            
            sim_time += self.dt
        
        # Train the agent
        if len(self.agent.memory) > 32:
            self.agent.replay(32)
        
        return episode_reward, steps, sim_time
    
    def train(self):
        print("Starting Neural RL Training with DQN...")
        print(f"Episodes: {self.num_episodes}")
        print(f"Episode duration: {self.episode_duration}s")
        print()
        
        try:
            for episode in range(self.num_episodes):
                # Regenerate obstacles periodically
                if episode > 0 and episode % 20 == 0:
                    self.world.generate_obstacles(20)
                
                # Run episode
                reward, steps, time_survived = self.run_episode(episode)
                self.episode_rewards.append(reward)
                self.episode_lengths.append(steps)
                
                # Update target network periodically
                if episode % 10 == 0:
                    self.agent.update_target_network()
                
                # Print progress
                recent_avg = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else reward
                print(f"Episode {episode}: Reward: {reward:.1f}, Steps: {steps}, "
                      f"Time: {time_survived:.1f}s, Epsilon: {self.agent.epsilon:.3f}, "
                      f"Avg10: {recent_avg:.1f}")
                
                # Save model periodically
                if (episode + 1) % 50 == 0:
                    torch.save(self.agent.q_network.state_dict(), f'dqn_model_episode_{episode+1}.pth')
                    print(f"Model saved at episode {episode+1}")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            # Save final model
            torch.save(self.agent.q_network.state_dict(), 'dqn_model_final.pth')
            print(f"\nTraining completed!")
            print(f"Average reward: {np.mean(self.episode_rewards):.2f}")
            print(f"Average steps: {np.mean(self.episode_lengths):.1f}")
            print(f"Final epsilon: {self.agent.epsilon:.3f}")
            
            self.visualizer.close()


def main():
    trainer = NeuralRLTrainer(
        num_episodes=200, 
        episode_duration=30.0, 
        max_velocity=300.0
    )
    trainer.train()


if __name__ == "__main__":
    main()