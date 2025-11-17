import torch
import torch.nn as nn
import numpy as np


class RobotNet(nn.Module):
    def __init__(self, hidden_size: int = 32, max_velocity: float = 300.0):
        super(RobotNet, self).__init__()
        
        # Input: 8 sonar readings (normalized 0-1) + 1 last action (one-hot encoded: 3 values)
        # Total input: 8 + 3 = 11
        # Hidden: 2 layers with Tanh
        # Output: 3 discrete actions (forward, correct_left, correct_right)
        
        self.fc1 = nn.Linear(11, hidden_size)  # Changed from 8 to 11
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 3)  # 3 actions
        
        self.tanh = nn.Tanh()
        self.max_velocity = max_velocity
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [batch, 11] = [batch, 8 sonar + 3 one-hot action]
        x = self.tanh(self.fc1(inputs))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)  # Raw logits for 3 actions
        
        return x
    
    def get_action(self, sonar_readings: np.ndarray, last_action: int) -> int:
        # Convert numpy array to tensor
        sonar_tensor = torch.FloatTensor(sonar_readings)
        
        # One-hot encode last action (3 values for 3 actions)
        last_action_onehot = torch.zeros(3)
        last_action_onehot[last_action] = 1.0
        
        # Concatenate sonar readings and last action
        input_tensor = torch.cat([sonar_tensor, last_action_onehot]).unsqueeze(0)
        
        with torch.no_grad():
            action_logits = self.forward(input_tensor)
        
        # Get action with highest score
        action = torch.argmax(action_logits, dim=1).item()
        
        # 0: forward, 1: correct_left, 2: correct_right
        return action
    
    def clone(self):
        new_model = RobotNet(hidden_size=self.fc1.out_features, max_velocity=self.max_velocity)
        new_model.load_state_dict(self.state_dict())
        return new_model
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.2):
        for param in self.parameters():
            if np.random.random() < mutation_rate:
                noise = torch.randn_like(param) * mutation_strength
                param.data += noise

