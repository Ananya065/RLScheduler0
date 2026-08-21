import os
import sys

torch_lib_path = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib_path):
    os.add_dll_directory(torch_lib_path)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.cpu_scheduler_env import CPUSchedulerEnv
from config import MODEL_SAVE_PATH

def evaluate():
    raw_env = DummyVecEnv([lambda: CPUSchedulerEnv(num_initial_jobs=100)])
    
    stats_path = "models/vec_normalize_phase1.pkl"
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, raw_env)
        env.training = False       
        env.norm_reward = False   
        env.norm_obs = True
    else:
        env = raw_env

    model = PPO.load(MODEL_SAVE_PATH, env=env)

    obs = env.reset()
    done = False

    wait_times = []

    while not done:
        unwrapped = env.envs[0]
        
        # Track waiting times of active processes in ready queue
        for job in unwrapped.ready_queue:
            wait_times.append(getattr(job, 'waiting_time', 0))
            
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

    unwrapped = env.envs[0]
    total_time = unwrapped.current_time
    avg_wait = (sum(wait_times) / len(wait_times)) if len(wait_times) > 0 else (total_time / 100)

    print("\n" + "="*45)
    print(f"PPO Agent Avg Waiting Time : {avg_wait:.2f} time units")
    print("="*45 + "\n")

if __name__ == "__main__":
    evaluate()
