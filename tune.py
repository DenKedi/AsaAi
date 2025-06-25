# tune.py

import os
import sys

sys.path.append(os.path.abspath('.'))

from train import train
import config


def tune():
    """Führt eine Suche über verschiedene Decay-Faktoren durch."""

    # --- Testen Sie verschiedene Geschwindigkeiten für den Epsilon-Abfall ---
    # Kleinerer Decay-Faktor = langsamerer Abfall
    decay_factors_to_test = [
        (config.EPS_START - config.EPS_END) / 50_000,  # Schneller Abfall
        (config.EPS_START - config.EPS_END) / 80_000,  # Mittlerer Abfall
        (config.EPS_START - config.EPS_END) / 120_000  # Langsamer Abfall
    ]

    results = {}

    for df in decay_factors_to_test:
        print(f"\n--- STARTE RUN: Decay Factor = {df:.8f} ---")

        final_score = train(decay_factor=df)

        print(f"--- ENDE RUN: End-Score = {final_score:.2f} ---")
        results[f"DF={df:.8f}"] = final_score

    print("\n\n--- TUNING ABGESCHLOSSEN ---")
    best_run = max(results, key=results.get)
    best_score = results[best_run]

    print(f"Beste Kombination: {best_run}")
    print(f"Bester Score (Avg Reward Last 50): {best_score:.2f}")


if __name__ == '__main__':
    tune()
