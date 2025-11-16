import torch
import torch.nn as nn
import numpy as np


class RobotNet(nn.Module):
    def __init__(self, hidden_size: int = 32, max_velocity: float = 300.0):
        super(RobotNet, self).__init__()
        
        # Input: 8 sonar readings (normalized 0-1)
        # Hidden: 2 layers with Tanh
        # Output: 2 values (left_vel, right_vel) with tanh → scale to [-max_velocity, max_velocity]
        
        self.fc1 = nn.Linear(8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        
        self.tanh = nn.Tanh()
        self.max_velocity = max_velocity
        
    def forward(self, sonar_inputs: torch.Tensor) -> torch.Tensor:
        x = self.tanh(self.fc1(sonar_inputs))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        
        # Scale from [-1, 1] to [-max_velocity, max_velocity]
        # Using reduced velocity during training for more stable learning
        x = x * self.max_velocity
        
        return x
    
    def get_action(self, sonar_readings: np.ndarray) -> tuple[float, float]:
        # Convert numpy array to tensor
        sonar_tensor = torch.FloatTensor(sonar_readings).unsqueeze(0)
        
        with torch.no_grad():
            velocities = self.forward(sonar_tensor)
        
        left_vel = velocities[0, 0].item()
        right_vel = velocities[0, 1].item()
        
        return left_vel, right_vel
    
    def clone(self):
        new_model = RobotNet(hidden_size=self.fc1.out_features, max_velocity=self.max_velocity)
        new_model.load_state_dict(self.state_dict())
        return new_model
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.2):
        for param in self.parameters():
            if np.random.random() < mutation_rate:
                noise = torch.randn_like(param) * mutation_strength
                param.data += noise

