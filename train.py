# train.py

import torch
import torch.optim as optim
import torch.nn as nn
import gymnasium as gym
import numpy as np
import random
import os

# Importieren unserer eigenen Module
import config
from agent.models import QNetwork
from agent.replay_buffer import ReplayBuffer


# Die Funktion akzeptiert jetzt Parameter für das Tuning.
# Wenn keine übergeben werden, nutzt sie die Werte aus config.py
def train(
        learning_rate=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        decay_factor=config.DECAY_FACTOR
):
    # --- Reproduzierbarkeit & Geräte-Setup ---
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    # ... (Rest des Setups bleibt gleich) ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # --- Agenten-Komponenten initialisieren ---
    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(config.BUFFER_SIZE)
    loss_fn = nn.SmoothL1Loss()

    # --- Epsilon-Setup für diesen spezifischen Lauf ---
    eps = config.EPS_START

    def get_epsilon_step(): #linear
        nonlocal eps  # Greift auf die 'eps'-Variable der äußeren train-Funktion zu
        eps = max(config.EPS_END, eps - decay_factor)
        return eps

    def getEpsilonExp():
        global eps
        eps = max(config.EPS_END, eps * decay_factor)
        return eps

    # --- Trainings-Loop ---
    global_step = 0
    episode_rewards = []

    for episode in range(config.MAX_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        #epsilon = get_epsilon_step() #epsilon linear decay per Episode (normal)

        for step in range(config.MAX_STEPS):
            global_step += 1
            epsilon = get_epsilon_step() #epsilon linear decay per Step (schnell)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax(1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= config.MIN_REPLAY_SIZE:
                states_s, actions_s, rewards_s, next_states_s, dones_s = replay_buffer.sample(batch_size, device)
                with torch.no_grad():
                    next_actions = policy_net(next_states_s).argmax(1, keepdim=True)
                    next_q_values = target_net(next_states_s).gather(1, next_actions)
                    target_q = rewards_s + (1 - dones_s) * config.GAMMA * next_q_values
                current_q = policy_net(states_s).gather(1, actions_s)
                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % config.TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        episode_rewards.append(total_reward)
        if (episode + 1) % 50 == 0:
            print(f"    Episode {episode + 1} - Avg Reward (last 50): {np.mean(episode_rewards[-50:]):.2f}")

    env.close()

    avg_reward_last_50 = np.mean(episode_rewards[-50:])
    return avg_reward_last_50


# Dieser Block erlaubt es, train.py auch direkt für einen einzelnen Lauf zu starten
if __name__ == '__main__':
    train()
