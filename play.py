import torch
import gymnasium as gym
import numpy as np
import os
import time

# Importieren unserer eigenen Module
import config
from agent.models import QNetwork


def play():
    # --- Geräte-Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benutztes Gerät: {device}")

    # --- Umgebung erstellen (mit 'human' Render-Modus für die Anzeige) ---
    env = gym.make(config.ENV_NAME, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # --- Modell laden ---
    if not os.path.exists(config.MODEL_PATH):
        print(f"Fehler: Modelldatei nicht gefunden unter {config.MODEL_PATH}")
        print("Bitte führen Sie zuerst train.py aus, um ein Modell zu trainieren.")
        return

    policy_net = QNetwork(state_dim, action_dim).to(device)
    policy_net.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    policy_net.eval()  # Wichtig: In den Evaluationsmodus schalten

    print("Modell geladen. Starte eine Demonstrations-Episode...")

    # --- Demonstrations-Schleife ---
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()  # Fenster anzeigen
        with torch.no_grad():
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)
            action = policy_net(state_tensor).argmax(1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        time.sleep(0.01)  # Kleine Pause, damit man zusehen kann

    print(f"Episode beendet. Erreichter Reward: {total_reward}")
    env.close()


if __name__ == '__main__':
    play()