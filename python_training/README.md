# Robot Neural Network Training

This directory contains multiple training approaches for the Pioneer robot:
1. **Data Collection & Supervised Learning** - Simple, interpretable
2. **Reinforcement Learning (PPO)** - More sample-efficient
3. **Evolutionary Algorithms** - Simpler, good for initial exploration

## Setup

1. Install Python dependencies:
```bash
cd python_training
pip install -r requirements.txt
```

## Training Options

### Option 1: Data Collection & Supervised Learning - RECOMMENDED FOR STARTING

Collect training data by testing all 4 directions at random positions:

```bash
python data_extractor.py
```

This will:
- Place robot at random positions in the world
- Test 4 directions (forward, backward, left, right) for 3 seconds each
- Evaluate which direction is best (no collision, most distance traveled)
- Visualize each test with colored trails
- Save dataset to `training_data.json`

**Dataset format:**
- Input: 8 normalized sonar readings [0-1]
- Output: Best direction index (0=forward, 1=backward, 2=left, 3=right)

**Advantages:**
- Simple and interpretable
- Fast data collection
- Clear ground truth labels
- Easy to visualize and debug

### Option 2: Reinforcement Learning (PPO)

Run the RL training script:
```bash
python train_rl.py
```

This will:
- Train a single actor-critic neural network using PPO
- Run 5 robots in parallel for faster experience collection
- Update policy every 2048 timesteps
- Visualize robots in real-time using Pygame
- Save the best model to `best_robot_rl.pth`
- Save checkpoints every 50 episodes

**Advantages:**
- More sample-efficient than evolutionary approach
- Learns faster with proper reward shaping
- Uses experience replay and value function estimation
- Better exploration-exploitation balance

### Option 3: Evolutionary Algorithm

Run the evolutionary training script:
```bash
python train.py
```

This will:
- Create a population of 10 neural network agents
- Simulate them in a 2D environment with obstacles
- Visualize all agents in real-time using Pygame
- Evolve the population over multiple generations (keep best 2, mutate, add random)
- Save the best model to `best_robot.pth`

**Advantages:**
- Simpler to understand and debug
- No gradient computation needed
- Good for initial exploration

## Setup

1. Install Python dependencies:
```bash
cd python_training
pip install -r requirements.txt
```

## Training Options

### Option 1: Reinforcement Learning (PPO) - RECOMMENDED

Run the RL training script:
```bash
python train_rl.py
```

This will:
- Train a single actor-critic neural network using PPO
- Run 5 robots in parallel for faster experience collection
- Update policy every 2048 timesteps
- Visualize robots in real-time using Pygame
- Save the best model to `best_robot_rl.pth`
- Save checkpoints every 50 episodes

**Advantages:**
- More sample-efficient than evolutionary approach
- Learns faster with proper reward shaping
- Uses experience replay and value function estimation
- Better exploration-exploitation balance

### Option 2: Evolutionary Algorithm

Run the evolutionary training script:
```bash
python train.py
```

This will:
- Create a population of 10 neural network agents
- Simulate them in a 2D environment with obstacles
- Visualize all agents in real-time using Pygame
- Evolve the population over multiple generations (keep best 2, mutate, add random)
- Save the best model to `best_robot.pth`

**Advantages:**
- Simpler to understand and debug
- No gradient computation needed
- Good for initial exploration

### Training Parameters

**For RL (train_rl.py):**
- `num_episodes`: Number of episodes to train (default: 1000)
- `episode_duration`: Duration of each episode in seconds (default: 30.0)
- `max_velocity`: Maximum velocity output (default: 300.0 mm/s)
- `num_parallel_robots`: Number of robots collecting experience simultaneously (default: 5)
- `update_frequency`: Update policy every N timesteps (default: 2048)
- PPO hyperparameters: learning_rate=3e-4, gamma=0.99, clip_epsilon=0.2

**For Evolutionary (train.py):**
- `num_generations`: Number of generations to train (default: 100)
- `episode_duration`: Duration of each episode in seconds (default: 30.0)
- `max_velocity`: Maximum velocity output (default: 300.0 mm/s)

### Reinforcement Learning (PPO)

Uses **Proximal Policy Optimization** with:
- **Actor-Critic architecture**: Shared feature layers, separate policy and value heads
- **Continuous action space**: Direct left/right wheel velocities
- **Reward function**:
  - +Forward movement reward (encourages progress)
  - +Exploration bonus (new grid cells)
  - -Collision penalty (-100)
  - -Danger penalty (when obstacles are close)
  - -Small time penalty (encourages efficiency)
- **GAE (Generalized Advantage Estimation)** for better value estimates
- **Experience replay** with parallel robots

### Evolutionary Algorithm

Each generation:
1. Run 10 agents for the episode duration
2. Calculate fitness based on distance traveled and exploration
3. Keep the top 2 performers
4. Create 2 mutated copies of the top 2
5. Generate 6 new random agents

## Exporting Weights

After training, export the weights for C++ inference:

**For RL-trained model:**
```bash
python export_weights_rl.py
```

**For evolutionary-trained model:**
```bash
python export_weights.py
```

Both create:
- `*.pth`: PyTorch model file
- `weights.json`: JSON format for C++ inference (only the actor/policy network)

## Model Architecture

**RL Model (Actor-Critic):**
- **Shared layers**: 8 inputs → 32 hidden (ReLU) → 32 hidden (ReLU)
- **Actor head**: Mean of action distribution (2 outputs with tanh) scaled to [-300, 300]
- **Critic head**: State value estimate (1 output)
- During inference, only the actor (policy) is exported and used

**Evolutionary Model:**
- **Input**: 8 sonar readings (normalized to 0-1)
- **Hidden**: 2 layers of 32 neurons with ReLU activation
- **Output**: 2 values (left_vel, right_vel) with tanh activation scaled to [-300, 300]

**Safety constraints:**
- Velocity is limited during training to prevent instability and teleporting
- Additional safety clamps in simulation: max 400 mm/s, max 50mm displacement per timestep

## Simulation Environment

- **World size**: 2000x2000 pixels
- **Obstacles**: 15 random rectangular obstacles per episode
- **Robot**: Differential drive with 8 sonar sensors
- **Sonar angles**: [90°, 50°, 30°, 10°, -10°, -30°, -50°, -90°]
- **Max sonar range**: 5000mm

## Reward/Fitness Functions

**RL Reward Function (per timestep):**
- Forward movement reward: `abs(velocity) × dt × 0.5` (significantly increased)
- Collision penalty: `-20` (further reduced from -50)
- Exploration bonus: `+3` for visiting new grid cells
- Danger penalty: `-0.1` only when obstacles very close (<300mm)
- **Spinning penalty**: 
  - `-0.1 × angular_velocity` only for severe spinning in place (forward speed < 30mm/s AND angular velocity > 1.5)
  - Much more lenient - allows turning while navigating
- **Forward movement bonus**: 
  - `+2.0` for moving forward (>50mm/s) with minimal rotation (<0.5 rad/s)
  - `+0.5` for moving forward even when turning
- **Survival bonus**: `+0.1` per timestep for staying alive (encourages survival)

**Evolutionary Fitness Function (per episode):**
- Fitness = `distance_traveled + (num_unique_cells_visited × 10)`
- Dead robots get 50% penalty
- Rewards forward movement and exploration

## Using the Trained Model in C++

1. Copy `weights.json` to the C++ project root:
```bash
cp python_training/weights.json .
```

2. Rebuild the C++ project:
```bash
make clean
make
```

3. Run the simulation:
```bash
./build/main
```

The C++ program will automatically load `weights.json` if it exists. If not found, it falls back to rule-based control.

## Visualization

During training, the Pygame window shows:
- All 10 agents simultaneously (different colors)
- Obstacles (gray rectangles)
- Sonar rays for the best agent (color-coded by distance)
- Real-time fitness scores
- Generation number and statistics

Press the window close button to stop training early.

