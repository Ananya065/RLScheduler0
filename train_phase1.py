# RLS/train_phase1.py
import os
import sys

# Force Windows/WSL DLL lookup paths
torch_lib_path = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib_path):
    os.add_dll_directory(torch_lib_path)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import Stable-Baselines3 wrappers
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.cpu_scheduler_env import CPUSchedulerEnv
from config import TOTAL_TIMESTEPS, MODEL_SAVE_PATH

def main():
    os.makedirs("models", exist_ok=True)
    
    # Wrap environment with VecNormalize to scale massive CPU rewards
    raw_env = DummyVecEnv([lambda: CPUSchedulerEnv(num_initial_jobs=100)])
    env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        vf_coef=0.01,          # Reduces value loss dominance
        max_grad_norm=0.5,     # Prevents exploding gradients
        tensorboard_log="./tensorboard_logs/"
    )

    print("Beginning Phase 1 PPO Training (with VecNormalize)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    # Save both model and environment normalization stats
    model.save(MODEL_SAVE_PATH)
    env.save("models/vec_normalize_phase1.pkl")
    print(f"Phase 1 model saved to {MODEL_SAVE_PATH}.zip")
    print("VecNormalize stats saved to models/vec_normalize_phase1.pkl")

if __name__ == "__main__":
    main()

