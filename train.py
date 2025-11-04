import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from qnetwork import QNetwork
from relaybuffer import ReplayBuffer


def train_dqn(
    env,
    device,
    obs_dim,
    act_dim,
    preprocess_obs=None,
    gamma=0.99,
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=5000,
    batch_size=64,
    buffer_size=10000,
    min_buffer=1000,
    target_update_freq=1000,
    training_episodes=1000,
    hidden_dim=64,
    use_epsilon_scheduler=True,   # Flag to use epsilon scheduler
    use_huber_loss=False        # Flag to use Huber loss
):
    policy_net = QNetwork(obs_dim, hidden_dim, act_dim).to(device)
    target_net = QNetwork(obs_dim, hidden_dim, act_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_size)
    all_rewards = []
    max_q_values = []

    total_steps = 0

    # Epsilon scheduling function
    def epsilon_by_step(step):
        return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-step / epsilon_decay)

    for episode in range(training_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_max_q = []

        while not done:
            total_steps += 1
            # Preprocess
            obs_t = torch.as_tensor(
                preprocess_obs(obs) if preprocess_obs else obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            # epsilon-greedy policy
            if use_epsilon_scheduler:
                epsilon = epsilon_by_step(total_steps)
            else:
                epsilon = epsilon_end  # constant epsilon

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(obs_t)
                    action = int(torch.argmax(q_values, dim=1).item())
                    episode_max_q.append(q_values.max().item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer
            if preprocess_obs:
                state_tensor = torch.as_tensor(preprocess_obs(obs), dtype=torch.float32, device=device)
                next_state_tensor = torch.as_tensor(preprocess_obs(next_obs), dtype=torch.float32, device=device)
                replay.push(state_tensor, action, reward, next_state_tensor, done)
            else:
                replay.push(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward

            # Train once replay buffer is large enough
            if len(replay) >= min_buffer:
                states, actions, rewards, next_states, dones = replay.sample(batch_size, device)

                # Compute targets
                with torch.no_grad():
                    next_q = target_net(next_states).max(1, keepdim=True)[0]
                    target_q = rewards + gamma * next_q * (1 - dones)

                # Compute predicted Q-values
                q_values = policy_net(states).gather(1, actions)

                # Choose loss based on flag
                if use_huber_loss:
                    loss = nn.functional.smooth_l1_loss(q_values, target_q)
                else:
                    loss = nn.functional.mse_loss(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Periodic target update
                if total_steps % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        all_rewards.append(episode_reward)
        max_q_values.append(np.mean(episode_max_q) if episode_max_q else 0.0)

        # Logging
        if episode % 100 == 0:
            print(f"Episode {episode:4d} | Reward: {episode_reward:6.1f} | ε={epsilon:.2f}")

    env.close()
    return policy_net, all_rewards, max_q_values
