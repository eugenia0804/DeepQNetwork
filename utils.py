import numpy as np
import torch

def preprocess_observation(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.sum(axis=2) # to greyscale
    mspacman_color = 210 + 164 + 74
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    return img.reshape(88, 80, 1)


def rollout_dqn(env, q_network, device, episodes=500, preprocess_obs=None, action_map=None):
    q_network.eval()
    rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Preprocess observation
            if preprocess_obs is not None:
                obs_t = preprocess_obs(obs).unsqueeze(0).to(device)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Greedy action selection (no epsilon)
            with torch.no_grad():
                q_values = q_network(obs_t)
                action = int(torch.argmax(q_values, dim=1).item())

            # Step the environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            obs = next_obs

        rewards.append(total_reward)

        if episode % 50 == 0:
            print(f"Episode {episode:4d} | Return: {total_reward:.1f}")

    env.close()
    return np.array(rewards)
