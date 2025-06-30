import pygame
import time
from stable_baselines3 import DQN

# Wir importieren die HulaHoopEnv-Klasse aus Ihrer Trainings-Datei.
from train_agent import HulaHoopEnv

# ==============================================================================
# INTERAKTIVES SPIEL MIT DER TRAINIERTEN KI (Version 2.0)
# ==============================================================================

# --- Interaktions-Parameter ---
# NEU: Separate Stärke für Anstupsen nach oben und unten.
#      Die Werte sind jetzt viel näher an der Spielphysik (z.B. KICK_POWER = 3.5).
NUDGE_DOWN_POWER = 2.5  # Stupst den Reifen nach unten (positive Geschwindigkeitsänderung)
NUDGE_UP_POWER = -2.5  # Stupst den Reifen nach oben (negative Geschwindigkeitsänderung)

# Schwellenwert, ab dem ein Eingriff möglich ist.
INTERVENTION_THRESHOLD_STEPS = 20

if __name__ == '__main__':
    # 1. Erstelle die Spielumgebung im sichtbaren Modus
    env = HulaHoopEnv(render_mode="human")

    # 2. Lade das trainierte Modell
    try:
        model = DQN.load("hula_dqn_model", env=env)
        print("Modell 'hula_dqn_model.zip' erfolgreich geladen.")
        print("\n--- STEUERUNG ---")
        print("Linksklick: Reifen sanft nach UNTEN stupsen.")
        print("Rechtsklick: Reifen sanft nach OBEN stupsen.")
        print("Eingriff erst nach 20Schritten möglich.")
        print("-----------------\n")

    except FileNotFoundError:
        print("Fehler: Die Datei 'hula_dqn_model.zip' wurde nicht gefunden.")
        exit()

    # 3. Endlos-Schleife für kontinuierliches Spielen
    while True:
        obs, info = env.reset()
        done = False
        episode_steps = 0

        print(f"\n--- Starte neue Episode ---")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if episode_steps > INTERVENTION_THRESHOLD_STEPS:
                        # NEU: Unterscheidung zwischen Links- und Rechtsklick
                        if event.button == 1:  # Linksklick
                            print(f"Eingriff bei Schritt {episode_steps}: Stupse nach UNTEN.")
                            env.hoop_y_velocity += NUDGE_DOWN_POWER
                        elif event.button == 3:  # Rechtsklick
                            print(f"Eingriff bei Schritt {episode_steps}: Stupse nach OBEN.")
                            env.hoop_y_velocity += NUDGE_UP_POWER
                    else:
                        print(f"Noch {INTERVENTION_THRESHOLD_STEPS - episode_steps} Schritte bis Eingriff möglich.")

            # KI wählt die nächste Aktion
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_steps += 1
            time.sleep(1 / 60)

        # Episode ist beendet
        reason = info.get("termination_reason", "Unbekannt")
        print(f"Episode nach {episode_steps} Schritten beendet. Grund: {reason}")
        time.sleep(1)