import torch
import json
import numpy as np
from train_rl_neural import SimpleQNetwork


def export_dqn_weights(model_path='dqn_model_final.pth', output_path='../weights.json'):
    """Export DQN model weights to JSON format for C++ usage"""
    
    # Load the trained model
    model = SimpleQNetwork(input_size=8, hidden_size=64, output_size=3)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Extract weights and biases
    weights_dict = {}
    
    # Layer 1: fc1 (8 -> 64)
    weights_dict['fc1_weight'] = model.fc1.weight.detach().numpy().tolist()
    weights_dict['fc1_bias'] = model.fc1.bias.detach().numpy().tolist()
    
    # Layer 2: fc2 (64 -> 64)  
    weights_dict['fc2_weight'] = model.fc2.weight.detach().numpy().tolist()
    weights_dict['fc2_bias'] = model.fc2.bias.detach().numpy().tolist()
    
    # Layer 3: fc3 (64 -> 3)
    weights_dict['fc3_weight'] = model.fc3.weight.detach().numpy().tolist()
    weights_dict['fc3_bias'] = model.fc3.bias.detach().numpy().tolist()
    
    # Add metadata
    weights_dict['metadata'] = {
        'input_size': 8,
        'hidden_size': 64,
        'output_size': 3,
        'activation': 'relu',
        'model_type': 'dqn',
        'description': 'Deep Q-Network trained for robot navigation'
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(weights_dict, f, indent=2)
    
    print(f"DQN weights exported to {output_path}")
    print(f"Model architecture: {8} -> {64} -> {64} -> {3}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test the model with dummy input
    dummy_input = torch.randn(1, 8)
    with torch.no_grad():
        output = model(dummy_input)
        action = torch.argmax(output, dim=1).item()
    
    print(f"Test input shape: {dummy_input.shape}")
    print(f"Test output shape: {output.shape}")
    print(f"Test predicted action: {action}")
    
    return weights_dict


def test_exported_weights(weights_path='../weights.json'):
    """Test the exported weights by loading and running inference"""
    
    with open(weights_path, 'r') as f:
        weights_dict = json.load(f)
    
    # Recreate model and load weights
    model = SimpleQNetwork(input_size=8, hidden_size=64, output_size=3)
    
    # Load weights manually
    model.fc1.weight.data = torch.tensor(weights_dict['fc1_weight'])
    model.fc1.bias.data = torch.tensor(weights_dict['fc1_bias'])
    model.fc2.weight.data = torch.tensor(weights_dict['fc2_weight'])
    model.fc2.bias.data = torch.tensor(weights_dict['fc2_bias'])
    model.fc3.weight.data = torch.tensor(weights_dict['fc3_weight'])
    model.fc3.bias.data = torch.tensor(weights_dict['fc3_bias'])
    
    model.eval()
    
    # Test with sample sonar readings
    test_sonar = [0.8, 0.9, 0.7, 0.6, 0.5, 0.7, 0.9, 0.8]  # Normalized sonar readings
    input_tensor = torch.FloatTensor(test_sonar).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        action = torch.argmax(output, dim=1).item()
    
    print(f"\nTest with sample sonar readings: {test_sonar}")
    print(f"Q-values: {output.numpy()[0]}")
    print(f"Predicted action: {action} ({'Forward' if action == 0 else 'Left' if action == 1 else 'Right'})")
    
    return True


if __name__ == "__main__":
    # Export the final trained model
    print("Exporting DQN weights...")
    export_dqn_weights('dqn_model_final.pth', '../weights.json')
    
    print("\nTesting exported weights...")
    test_exported_weights('../weights.json')
    
    print("\nExport completed successfully!")