# RLS/config.py

# Queue & Environment Limits
MAX_QUEUE_SIZE = 50
MAX_BURST_TIME = 20.0
CONTEXT_SWITCH_COST = 0.5

# Reward Weight Factors (Turnaround & Waiting Optimization)
ALPHA = 1.0  # Waiting time penalty multiplier
BETA = 0.5   # Turnaround time penalty multiplier
GAMMA = 0.2  # Context switch overhead penalty multiplier

# Training Settings
TOTAL_TIMESTEPS = 100000
MODEL_SAVE_PATH = "models/phase1_ppo_cpu_scheduler"