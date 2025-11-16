import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import List, Tuple


class RobotDataset(Dataset):
    def __init__(self, data_file: str, weight_by_distance: bool = True):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        # Direction to velocity mapping
        self.direction_velocities = {
            0: (300.0, 300.0),   # forward
            1: (250.0, 300.0),   # forward_left
            2: (300.0, 250.0),   # forward_right
        }
        
        # Calculate sample weights based on distance traveled
        self.weights = None
        if weight_by_distance:
            distances = []
            for sample in self.data:
                best_dir_name = sample['best_direction_name']
                distance = sample['results'][best_dir_name]['distance_traveled']
                distances.append(max(0.0, distance))  # Ensure non-negative
            
            # Normalize weights to sum to len(data) (so average weight = 1.0)
            if distances:
                total_distance = sum(distances)
                if total_distance > 0:
                    self.weights = [d / total_distance * len(self.data) for d in distances]
                else:
                    self.weights = [1.0] * len(self.data)
            else:
                self.weights = [1.0] * len(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Input: sensor readings (8 values)
        sensor_readings = torch.FloatTensor(sample['sensor_readings'])
        
        # Output: velocities for the best direction
        best_dir = sample['best_direction']
        left_vel, right_vel = self.direction_velocities[best_dir]
        velocities = torch.FloatTensor([left_vel, right_vel])
        
        # Return weight if available
        if self.weights:
            weight = torch.FloatTensor([self.weights[idx]])
            return sensor_readings, velocities, weight
        
        return sensor_readings, velocities


class SupervisedRobotNet(nn.Module):
    def __init__(self, hidden_size: int = 32, max_velocity: float = 300.0):
        super(SupervisedRobotNet, self).__init__()
        
        # Input: 8 sonar readings (normalized 0-1)
        # Hidden: 2 layers with ReLU
        # Output: 2 values (left_vel, right_vel) with tanh → scale to [-max_velocity, max_velocity]
        
        self.fc1 = nn.Linear(8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.max_velocity = max_velocity
        
    def forward(self, sonar_inputs: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(sonar_inputs))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        
        # Scale from [-1, 1] to [-max_velocity, max_velocity]
        x = x * self.max_velocity
        
        return x


def train_supervised_model(data_file: str = 'training_data.json', 
                          num_epochs: int = 100,
                          batch_size: int = 32,
                          learning_rate: float = 0.001,
                          weight_by_distance: bool = True):
    """Train a supervised learning model on collected data
    
    Args:
        weight_by_distance: If True, weight samples by distance traveled (optimizes for distance)
    """
    
    print("Loading dataset...")
    dataset = RobotDataset(data_file, weight_by_distance=weight_by_distance)
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create samplers with weights if enabled
    if weight_by_distance and dataset.weights:
        train_indices = train_dataset.indices
        train_weights = [dataset.weights[i] for i in train_indices]
        train_sampler = torch.utils.data.WeightedRandomSampler(
            train_weights, num_samples=len(train_indices), replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    if weight_by_distance and dataset.weights:
        avg_weight = sum(dataset.weights) / len(dataset.weights)
        max_weight = max(dataset.weights)
        min_weight = min(dataset.weights)
        print(f"Sample weights: avg={avg_weight:.2f}, min={min_weight:.2f}, max={max_weight:.2f}")
    
    # Create model
    model = SupervisedRobotNet(hidden_size=32, max_velocity=300.0)
    criterion = nn.MSELoss(reduction='none')  # Use 'none' to apply weights per sample
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            
            if weight_by_distance and len(batch) == 3:
                sensor_inputs, target_velocities, weights = batch
                weights = weights.squeeze()
            else:
                sensor_inputs, target_velocities = batch
                weights = None
            
            # Forward pass
            predicted_velocities = model(sensor_inputs)
            
            # Loss per sample
            sample_losses = criterion(predicted_velocities, target_velocities)
            # Average over the 2 outputs (left_vel, right_vel)
            sample_losses = sample_losses.mean(dim=1)
            
            # Apply weights if available
            if weights is not None:
                loss = (sample_losses * weights).mean()
            else:
                loss = sample_losses.mean()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation (no weighting)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    sensor_inputs, target_velocities, _ = batch
                else:
                    sensor_inputs, target_velocities = batch
                
                predicted_velocities = model(sensor_inputs)
                sample_losses = criterion(predicted_velocities, target_velocities)
                loss = sample_losses.mean()
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_robot_supervised.pth')
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print("Model saved to 'best_robot_supervised.pth'")
    
    # Load best model and test
    model.load_state_dict(torch.load('best_robot_supervised.pth'))
    model.eval()
    
    # Test on a few samples
    print("\nTesting on sample inputs:")
    with torch.no_grad():
        for i in range(min(5, len(val_dataset))):
            batch = val_dataset[i]
            if len(batch) == 3:
                sensor_inputs, target_velocities, _ = batch
            else:
                sensor_inputs, target_velocities = batch
            
            predicted_velocities = model(sensor_inputs.unsqueeze(0))
            
            print(f"\nSample {i+1}:")
            print(f"  Sensor: {sensor_inputs.numpy()}")
            print(f"  Target:  Left={target_velocities[0]:.1f}, Right={target_velocities[1]:.1f}")
            print(f"  Predicted: Left={predicted_velocities[0,0]:.1f}, Right={predicted_velocities[0,1]:.1f}")
    
    return model


def export_supervised_model_to_json(model: SupervisedRobotNet, output_path: str):
    """Export supervised model to JSON format for C++ inference"""
    import json
    
    model.eval()
    
    # Get architecture
    architecture = [
        model.fc1.in_features,
        model.fc1.out_features,
        model.fc2.out_features,
        model.fc3.out_features
    ]
    
    # Extract weights and biases
    layers = []
    
    # Layer 1
    layers.append({
        "weights": model.fc1.weight.detach().cpu().numpy().tolist(),
        "biases": model.fc1.bias.detach().cpu().numpy().tolist()
    })
    
    # Layer 2
    layers.append({
        "weights": model.fc2.weight.detach().cpu().numpy().tolist(),
        "biases": model.fc2.bias.detach().cpu().numpy().tolist()
    })
    
    # Layer 3
    layers.append({
        "weights": model.fc3.weight.detach().cpu().numpy().tolist(),
        "biases": model.fc3.bias.detach().cpu().numpy().tolist()
    })
    
    # Create output dictionary
    output = {
        "architecture": architecture,
        "layers": layers,
        "max_velocity": model.max_velocity
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nModel exported to {output_path}")
    print(f"Architecture: {architecture}")
    print(f"Max velocity: {model.max_velocity}")


def main():
    # Train model with distance-based weighting (optimizes for distance traveled)
    model = train_supervised_model(
        data_file='training_data.json',
        num_epochs=300,
        batch_size=32,
        learning_rate=0.001,
        weight_by_distance=True  # Weight samples by distance traveled
    )
    
    # Export to JSON
    export_supervised_model_to_json(model, 'weights.json')
    
    print("\nDone! Model ready for C++ inference.")


if __name__ == "__main__":
    main()

