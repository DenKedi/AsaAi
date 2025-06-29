import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env


# ==============================================================================
# SCHRITT 3: DIE GYMNASIUM-UMGEBUNG
# ==============================================================================
class HulaHoopEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode=None):
        super().__init__()

        # === NEU: Definiere Aktions- und Beobachtungsraum ===
        # Aktion: 0 (nichts tun) oder 1 (Schwung geben)
        self.action_space = spaces.Discrete(2)

        # Beobachtung: Ein Vektor mit 3 Werten:
        # [hoop_y, hoop_y_velocity, swing_speed]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        # === DEIN SPIELCODE, JETZT TEIL DER KLASSE ===
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

        # Pygame-spezifische Variablen
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

    def _get_obs(self):
        """Gibt die aktuelle Beobachtung zurück."""
        return np.array([self.hoop_y, self.hoop_y_velocity, self.swing_speed], dtype=np.float32)

    def _get_info(self):
        """Gibt zusätzliche Informationen zurück (optional)."""
        return {"swing_speed": self.swing_speed}

    def reset(self, seed=None, options=None):
        """Setzt das Spiel auf den Anfangszustand zurück."""
        super().reset(seed=seed)

        self.hoop_y = self.robot_y + self.robot_height / 2
        self.hoop_y_velocity = 0.0
        self.swing_speed = 8.0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        """Führt einen Spielschritt aus."""
        # --- Aktion ausführen ---
        if action == 1:
            self.swing_speed += self.SWING_BOOST
            self.hoop_y_velocity -= self.KICK_POWER

        # --- Physik-Update (dein Code) ---
        self.swing_speed *= self.SWING_DECAY
        lift_force = self.swing_speed * self.LIFT_FACTOR
        if self.swing_speed > self.SWEET_SPOT_MAX:
            lift_force += (self.swing_speed - self.SWEET_SPOT_MAX) * self.UPWARD_FLIGHT_FACTOR
        self.hoop_y_velocity += self.GRAVITY - lift_force
        self.hoop_y += self.hoop_y_velocity

        # --- Zustand prüfen und Belohnung (Reward) definieren ---
        terminated = self.hoop_y > self.GAME_OVER_BOTTOM or self.hoop_y < self.GAME_OVER_TOP

        # ==========================================================
        # NEUE, BESSERE REWARD-LOGIK
        # ==========================================================
        reward = 0
        if terminated:
            reward = -100  # Große Strafe für's Verlieren bleibt
        else:
            # Belohnung dafür, in der Nähe des Zentrums zu sein
            sweet_spot_center = self.robot_y + self.robot_height / 2
            distance_to_center = abs(self.hoop_y - sweet_spot_center)

            # Die Belohnung ist höher, je näher der Reifen am Zentrum ist.
            # Max. Belohnung = 2.0, Min. Belohnung = 0
            # Die 80 ist hier die Höhe der "Sweet Spot Zone" aus dem alten Code.
            max_distance = 80
            if distance_to_center < max_distance:
                # Linearer Bonus: 2.0 wenn perfekt im Zentrum, 0 am Rand der Zone.
                reward = 2.0 * (1 - (distance_to_center / max_distance))

            # Kleine "Überlebensbelohnung" kann man zusätzlich geben, ist aber optional
            reward += 0.1
        # ==========================================================

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, self._get_info()
    def render(self):
        """Diese Methode wird für das Anzeigen des Spiels aufgerufen."""
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        """Enthält deine gesamte Zeichen-Logik."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Hula Hoop Roboter - KI Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        # Dein gesamter Zeichen-Code
        self.screen.fill(self.BACKGROUND_COLOR)
        pygame.draw.rect(self.screen, self.ROBOT_COLOR, self.robot_rect)
        pygame.draw.rect(self.screen, self.ROBOT_COLOR, self.robot_head_rect)

        hoop_width = 150 + (self.hoop_y - self.robot_y) * 0.1
        hoop_rect = pygame.Rect(self.robot_x + self.robot_width / 2 - hoop_width / 2, self.hoop_y, hoop_width, 25)
        pygame.draw.ellipse(self.screen, self.HOOP_COLOR, hoop_rect, 6)

        swing_text = self.font.render(f"Schwung: {self.swing_speed:.1f}", True, self.WHITE)
        self.screen.blit(swing_text, (10, 10))

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Räumt Pygame auf, wenn die Umgebung geschlossen wird."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


# ==============================================================================
# SCHRITT 4: AGENTEN ERSTELLEN UND TRAINIEREN
# ==============================================================================
if __name__ == '__main__':
    # 1. Erstelle eine Instanz der Umgebung
    env = HulaHoopEnv()

    # Optional: Prüfe, ob deine Umgebung dem Gymnasium-Standard entspricht
    check_env(env)

    # 2. Definiere die Architektur des neuronalen Netzes
    policy_kwargs = dict(
        net_arch=[128, 128]  # Zwei versteckte Schichten mit je 128 Neuronen
    )

    # 3. Erstelle das DQN-Modell
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        learning_rate=1e-4,  # 0.0001
        tensorboard_log="./hula_dqn_tensorboard/",
        exploration_fraction=0.2,
        exploration_initial_eps=0.7,
        exploration_final_eps=0.02
    )

    # 4. Trainiere das Modell
    print("Beginne mit dem Training...")
    model.learn(total_timesteps=170000, progress_bar=True)

    # 5. Speichere das trainierte Modell
    model.save("hula_dqn_model")
    print("Training abgeschlossen. Modell wurde als 'hula_dqn_model.zip' gespeichert.")

    env.close()