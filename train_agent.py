import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from collections import Counter
import sys
import torch  # Wichtig für die neue Logik
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import time


# Die HulaHoopEnv-Klasse bleibt unverändert.
# (Sie können hier Ihre bestehende Klasse einfügen)
class HulaHoopEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    # ... (kompletter Code Ihrer HulaHoopEnv-Klasse hier)
    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32)
        self.hoop_y = 0.0
        self.hoop_y_velocity = 0.0
        self.swing_speed = 0.0
        self.current_episode_steps = 0
        self.current_episode_reward = 0.0
        self.last_action = 0
        self.last_reward = 0.0
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.BACKGROUND_COLOR = (20, 30, 40)
        self.WHITE = (255, 255, 255)
        self.ROBOT_COLOR = (130, 140, 150)
        self.HOOP_COLOR = (255, 105, 180)
        self.robot_width = 80
        self.robot_height = 200
        self.robot_x = (self.SCREEN_WIDTH - self.robot_width) / 2
        self.robot_y = self.SCREEN_HEIGHT - self.robot_height - 20
        self.robot_rect = pygame.Rect(self.robot_x, self.robot_y, self.robot_width, self.robot_height)
        self.robot_head_rect = pygame.Rect(self.robot_x - 10, self.robot_y - 40, self.robot_width + 20, 40)
        self.SWING_DECAY = 0.99
        self.SWING_BOOST = 2.0
        self.GRAVITY = 0.15
        self.KICK_POWER = 3.5
        self.LIFT_FACTOR = 0.01
        self.UPWARD_FLIGHT_FACTOR = 0.04
        self.SWEET_SPOT_MIN = 12.0
        self.SWEET_SPOT_MAX = 22.0
        self.GAME_OVER_BOTTOM = self.SCREEN_HEIGHT - 30
        self.GAME_OVER_TOP = self.robot_y - 20
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

    def _get_obs(self):
        return np.array([self.hoop_y, self.hoop_y_velocity, self.swing_speed], dtype=np.float32)

    def _get_info(self):
        return {"swing_speed": self.swing_speed}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        sweet_spot_center = self.robot_y + self.robot_height / 2
        self.hoop_y = sweet_spot_center + self.np_random.uniform(-20, 20)
        self.swing_speed = self.np_random.uniform(7.0, 10.0)
        self.hoop_y_velocity = 0.0
        self.current_episode_steps = 0
        self.current_episode_reward = 0.0
        if self.render_mode == "human": self._render_frame()
        return self._get_obs(), self._get_info()

    def step(self, action):
        if action == 1:
            self.swing_speed += self.SWING_BOOST
            self.hoop_y_velocity -= self.KICK_POWER
        self.swing_speed *= self.SWING_DECAY
        lift_force = self.swing_speed * self.LIFT_FACTOR
        if self.swing_speed > self.SWEET_SPOT_MAX:
            lift_force += (self.swing_speed - self.SWEET_SPOT_MAX) * self.UPWARD_FLIGHT_FACTOR
        self.hoop_y_velocity += self.GRAVITY - lift_force
        self.hoop_y += self.hoop_y_velocity
        terminated = self.hoop_y > self.GAME_OVER_BOTTOM or self.hoop_y < self.GAME_OVER_TOP
        info = self._get_info()
        reward = 0
        if terminated:
            reward = -100
            if self.hoop_y > self.GAME_OVER_BOTTOM:
                info["termination_reason"] = "Gefallen"
            else:
                info["termination_reason"] = "Hochgeflogen"
        else:
            sweet_spot_center = self.robot_y + self.robot_height / 2
            distance_to_center = abs(self.hoop_y - sweet_spot_center)
            max_distance = 80
            if distance_to_center < max_distance:
                reward = 2.0 * (1 - (distance_to_center / max_distance))
            reward -= abs(self.hoop_y_velocity) * 0.1
        self.last_action = action
        self.last_reward = reward
        self.current_episode_steps += 1
        self.current_episode_reward += reward
        if self.render_mode == "human": self._render_frame()
        return self._get_obs(), reward, terminated, False, info

    def render(self):
        if self.render_mode == "human": return self._render_frame()

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Hula Hoop Roboter - KI Interaktion")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 28)
        self.screen.fill(self.BACKGROUND_COLOR)
        pygame.draw.rect(self.screen, self.ROBOT_COLOR, self.robot_rect)
        pygame.draw.rect(self.screen, self.ROBOT_COLOR, self.robot_head_rect)
        hoop_width = 150 + (self.hoop_y - self.robot_y) * 0.1
        hoop_rect = pygame.Rect(self.robot_x + self.robot_width / 2 - hoop_width / 2, self.hoop_y, hoop_width, 25)
        pygame.draw.ellipse(self.screen, self.HOOP_COLOR, hoop_rect, 6)
        hud_texts = ["--- AGENTEN-STATUS ---", f"Schwung: {self.swing_speed:.2f}", f"Position Y: {self.hoop_y:.2f}",
                     f"Geschwindigkeit Y: {self.hoop_y_velocity:.2f}", "--- AKTION & BELOHNUNG ---",
                     f"Letzte Aktion: {'Schwung geben' if self.last_action == 1 else 'Nichts tun'}",
                     f"Frame-Belohnung: {self.last_reward:.2f}", "--- EPISODEN-STATUS ---",
                     f"Schritte: {self.current_episode_steps}", f"Gesamtbelohnung: {self.current_episode_reward:.2f}"]
        for i, text in enumerate(hud_texts):
            text_surface = self.font.render(text, True, self.WHITE)
            self.screen.blit(text_surface, (10, 10 + i * 25))
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


# ==============================================================================
# NEUE KLASSE: DoubleDQN
# ==============================================================================
class DoubleDQN(DQN):
    # Wir erben alles von der originalen DQN-Klasse und überschreiben nur die train-Methode
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Standard-Setup aus der Original-Methode
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # Log-Variablen
        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # ==========================================================
                # HIER IST DER ENTSCHEIDENDE UNTERSCHIED ZU NORMALEM DQN
                # ==========================================================

                # --- Originale DQN-Logik (jetzt auskommentiert) ---
                # # Compute the next Q-values using the target network
                # next_q_values_target = self.q_net_target(replay_data.next_observations)
                # # Follow greedy policy: use the one with the highest value
                # next_q_values, _ = next_q_values_target.max(dim=1)

                # --- Neue Double DQN-Logik ---
                # 1. Wähle die beste Aktion mit dem Online-Netzwerk (self.q_net) aus
                next_q_values_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_values_online.argmax(dim=1)

                # 2. Evaluiere den Wert dieser Aktion mit dem Target-Netzwerk (self.q_net_target)
                next_q_values_target = self.q_net_target(replay_data.next_observations)
                # .gather() wählt gezielt die Werte aus, die zu den von 'next_actions' bestimmten Indizes gehören
                next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)

                # ==========================================================

                # Berechnung des Ziel-Q-Wertes (1-step TD target)
                # .flatten() sorgt dafür, dass die Dimensionen stimmen
                target_q_values = replay_data.rewards.flatten() + (
                            1 - replay_data.dones.flatten()) * self.gamma * next_q_values

            # Berechne die aktuellen Q-Werte mit dem Online-Netzwerk
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Berechne den Loss (Verlust)
            loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
            losses.append(loss.item())

            # Optimierungsschritt
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Update der internen Zähler und Logs
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


# ==============================================================================
# HAUPTTEIL
# ==============================================================================
if __name__ == '__main__':
    # ... (hier könnten Sie die Multi-Run-Logik einfügen) ...

    # Erstelle die Umgebung
    train_env = HulaHoopEnv()

    # Definiere Hyperparameter
    policy_kwargs = dict(net_arch=[256, 256])

    # ==========================================================
    # VERWENDE JETZT UNSERE NEUE DoubleDQN-KLASSE
    # ==========================================================
    model = DoubleDQN(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        buffer_size=200000,
        learning_starts=10000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        learning_rate=5e-5,
        tau=0.005,
        tensorboard_log="./hula_double_dqn_tensorboard/",  # Eigener Log-Ordner
        exploration_fraction=0.2,
        exploration_initial_eps=0.7,
        exploration_final_eps=0.02
    )

    print("Beginne mit dem Training des DoubleDQN-Modells...")
    model.learn(total_timesteps=170000, progress_bar=True)
    model.save("hula_double_dqn_model")
    print("Training abgeschlossen. Modell wurde als 'hula_double_dqn_model.zip' gespeichert.")
    train_env.close()
