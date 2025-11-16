import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from model import RobotNet


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            value: float, log_prob: float, done: bool):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self) -> Tuple[torch.Tensor, ...]:
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        rewards = torch.FloatTensor(self.rewards)
        values = torch.FloatTensor(self.values)
        log_probs = torch.FloatTensor(self.log_probs)
        dones = torch.FloatTensor(self.dones)
        
        return states, actions, rewards, values, log_probs, dones


class ActorCriticNet(nn.Module):
    def __init__(self, hidden_size: int = 32, max_velocity: float = 300.0):
        super(ActorCriticNet, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_size, 2)
        self.actor_log_std = nn.Parameter(torch.zeros(2))
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.max_velocity = max_velocity
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        
        # Actor: output mean of action distribution
        action_mean = self.tanh(self.actor_mean(x)) * self.max_velocity
        
        # Critic: output state value
        value = self.critic(x)
        
        return action_mean, value
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_mean, value = self.forward(state_tensor)
        
        if deterministic:
            action = action_mean
        else:
            # Sample from Gaussian distribution
            std = torch.exp(self.actor_log_std)
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        if deterministic:
            return action[0].numpy(), 0.0, value[0].item()
        else:
            return action[0].numpy(), log_prob.item(), value[0].item()
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, values = self.forward(states)
        
        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(action_mean, std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, values.squeeze(), entropy


class PPOTrainer:
    def __init__(self, model: ActorCriticNet, learning_rate: float = 3e-4, 
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, epochs: int = 10, 
                 batch_size: int = 64):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.buffer = RolloutBuffer()
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                    dones: torch.Tensor, next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        values_with_next = torch.cat([values, torch.tensor([next_value])])
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values_with_next[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, next_value: float) -> dict:
        states, actions, rewards, values, old_log_probs, dones = self.buffer.get()
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.epochs):
            # Generate random indices for mini-batches
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Evaluate current policy
                log_probs, state_values, entropy = self.model.evaluate_actions(batch_states, batch_actions)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(state_values, batch_returns)
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / self.epochs,
            'value_loss': total_value_loss / self.epochs,
            'entropy': total_entropy / self.epochs
        }


