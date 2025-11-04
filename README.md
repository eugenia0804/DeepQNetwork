# CartPole & MsPacman Reinforcement Learning with Deep Q Network

This project implements a DQN algorithm to train an agent to play CartPole and MsPacman games.

## Project Structure

- `train.py`: Main training loop
- `utils.py`: Data processing and network rollout utilities
- `vis.py`: Visualization utilities
- `qnetwork.py`: Architecture for the Q-Network
- `relaybuffer.py`: Implementation for the Relay BUffer

## Requirements

```bash
pip install torch gymnasium ale-py matplotlib numpy
```

## Components

### Q-Network Architecture (`qnetwork.py`)

A multi-layer perceptron (MLP) that approximates the Q-function, mapping input states to action-value estimates using ReLU activations.

### Relay Buffer Implementation (`relaybuffer.py`)

An experience replay mechanism that stores transitions and samples random batches to stabilize DQN training by breaking temporal correlations.

### DQN Algorithm (`train.py`)

Thee core Deep Q-Network (DQN) training pipeline intergrats environment interaction, experience replay, and target network updates. The agent follows an epsilon-greedy policy to balance exploration and exploitation, storing transitions in a replay buffer to decorrelate training samples. Each step, when the buffer is sufficiently filled, a batch of experiences is sampled to compute the TD target using the frozen target network. The Q-Network is then optimized via gradient descent to minimize the MSE (or Huber) loss between predicted and target Q-values. The target network is periodically synchronized with the Q-Network to stabilize learning. 

### Advance Training Techniques (`train.py`)

An epsilon scheduler adaptively decays exploration over time using an exponential decay function to enable efficient Q-network refinement as the agent learns. The option to use Huber loss provides robustness against outliers and reduces gradient sensitivity during unstable early training.

### Trained Network Rollout Techniques (`utils.py`)

Execution of the evaluation rollouts of a trained DQN is in a deterministic (greedy) manner to measure real performance without exploration noise. For each episode, the agent repeatedly selects the highest-Q action from the network’s predictions, interacts with the environment, and accumulates total rewards. 

### Result Visualization

Detailed visualization and analysis are available in `res_cartpole.ipynb` and `res_mspacman.ipynb`.

