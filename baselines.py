# RLS/baselines.py
import os
import sys
import ctypes

# 1. Force Windows to resolve dynamic library dependencies in site-packages
torch_lib_path = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")

if os.path.exists(torch_lib_path):
    os.add_dll_directory(torch_lib_path)
    # Pre-load OpenMP / Intel MKL libraries directly into process memory space
    for dll_name in ["libiomp5md.dll", "fbjemm.dll", "c10.dll"]:
        dll_file = os.path.join(torch_lib_path, dll_name)
        if os.path.exists(dll_file):
            try:
                ctypes.CDLL(dll_file)
            except Exception:
                pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Import PyTorch & SB3 ---
import torch
from stable_baselines3 import PPO
import copy
from workload_generator import generate_synthetic_workload

def run_round_robin(workload, quantum=2.0):
    jobs = copy.deepcopy(workload)
    ready_queue = []
    current_time = 0.0
    total_waiting_time = 0.0
    completed = 0
    num_jobs = len(jobs)
    
    while completed < num_jobs:
        while jobs and jobs[0].arrival_time <= current_time:
            ready_queue.append(jobs.pop(0))
            
        if not ready_queue:
            current_time += 1.0
            continue
            
        proc = ready_queue.pop(0)
        exec_time = min(quantum, proc.remaining_time)
        proc.remaining_time -= exec_time
        current_time += exec_time
        
        # Update waiting time for remaining queued items
        for p in ready_queue:
            p.waiting_time += exec_time
            
        while jobs and jobs[0].arrival_time <= current_time:
            ready_queue.append(jobs.pop(0))
            
        if proc.remaining_time > 0:
            ready_queue.append(proc)
        else:
            total_waiting_time += proc.waiting_time
            completed += 1

    return total_waiting_time / num_jobs

if __name__ == "__main__":
    test_trace = generate_synthetic_workload(50)
    avg_wait_rr = run_round_robin(test_trace, quantum=2.0)
    print(f"Round Robin (Quantum=2.0) Avg Waiting Time: {avg_wait_rr:.2f} time units")