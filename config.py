# config.py

# --- Umgebungsparameter ---
ENV_NAME = "CartPole-v1"
SEED = 1
MODEL_PATH = "saved_models/dqn_cartpole.pth"

# --- Trainings-Parameter ---
MAX_EPISODES = 500
MAX_STEPS = 500
LEARNING_RATE = 0.00025
GAMMA = 0.99
TARGET_UPDATE_FREQ = 1000

# --- Replay Buffer & Batch ---
BUFFER_SIZE = 100_000
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 1_000

# --- Epsilon-Greedy-Parameter (Ihre Original-Logik) ---
EPS_START = 0.7 
EPS_END = 0.05

# Standard-Decay-Faktor, wenn train.py direkt ausgeführt wird.
# Wird von tune.py überschrieben.
# Annahme: Abfall über ca. 80.000 Schritte
DECAY_FACTOR = (EPS_START - EPS_END) / 80_000 
