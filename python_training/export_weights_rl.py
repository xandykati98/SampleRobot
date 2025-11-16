import torch
import json
import numpy as np
from rl_trainer import ActorCriticNet


def export_actor_to_json(model: ActorCriticNet, output_path: str):
    """
    Export only the actor (policy) part of the ActorCritic model to JSON for C++ inference.
    We extract the shared layers + actor mean to create a simple feedforward network.
    """
    
    model.eval()
    
    # Architecture: [8, 32, 32, 2]
    architecture = [
        model.fc1.in_features,
        model.fc1.out_features,
        model.fc2.out_features,
        2  # Actor output
    ]
    
    # Extract weights and biases
    layers = []
    
    # Layer 1 (shared)
    layers.append({
        "weights": model.fc1.weight.detach().cpu().numpy().tolist(),
        "biases": model.fc1.bias.detach().cpu().numpy().tolist()
    })
    
    # Layer 2 (shared)
    layers.append({
        "weights": model.fc2.weight.detach().cpu().numpy().tolist(),
        "biases": model.fc2.bias.detach().cpu().numpy().tolist()
    })
    
    # Layer 3 (actor mean only - deterministic policy for deployment)
    layers.append({
        "weights": model.actor_mean.weight.detach().cpu().numpy().tolist(),
        "biases": model.actor_mean.bias.detach().cpu().numpy().tolist()
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
    
    print(f"Actor network exported to {output_path}")
    print(f"Architecture: {architecture}")
    print(f"Max velocity: {model.max_velocity}")


def main():
    # Load the trained model
    model = ActorCriticNet(hidden_size=32, max_velocity=300.0)
    
    try:
        model.load_state_dict(torch.load('best_robot_rl.pth'))
        print("Loaded best_robot_rl.pth")
    except FileNotFoundError:
        print("Error: best_robot_rl.pth not found. Please train the model first.")
        return
    
    # Export to JSON
    export_actor_to_json(model, 'weights.json')
    
    # Test the model with sample input
    print("\nTesting model with sample input:")
    sample_input = torch.FloatTensor([[0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5]])
    with torch.no_grad():
        action_mean, value = model(sample_input)
    print(f"Input: {sample_input.numpy()}")
    print(f"Output (left_vel, right_vel): {action_mean.numpy()}")
    print(f"State Value: {value.item():.3f}")


if __name__ == "__main__":
    main()


