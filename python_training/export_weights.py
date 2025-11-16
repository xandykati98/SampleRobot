import torch
import json
import numpy as np
from model import RobotNet


def export_model_to_json(model: RobotNet, output_path: str):
    """
    Export PyTorch model weights to JSON format for C++ inference.
    
    Format:
    {
        "architecture": [8, 32, 32, 2],
        "layers": [
            {
                "weights": [[...], [...], ...],
                "biases": [...]
            },
            ...
        ]
    }
    """
    
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
        "layers": layers
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Model exported to {output_path}")
    print(f"Architecture: {architecture}")


def main():
    # Load the trained model
    # Note: max_velocity should match the training configuration
    model = RobotNet(hidden_size=32, max_velocity=300.0)
    
    try:
        model.load_state_dict(torch.load('best_robot.pth'))
        print("Loaded best_robot.pth")
        print(f"Model max velocity: {model.max_velocity}")
    except FileNotFoundError:
        print("Error: best_robot.pth not found. Please train the model first.")
        return
    
    # Export to JSON
    export_model_to_json(model, 'weights.json')
    
    # Test the model with sample input
    print("\nTesting model with sample input:")
    sample_input = torch.FloatTensor([[0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5]])
    with torch.no_grad():
        output = model(sample_input)
    print(f"Input: {sample_input.numpy()}")
    print(f"Output (left_vel, right_vel): {output.numpy()}")


if __name__ == "__main__":
    main()

