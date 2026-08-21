# RLS/workload_generator.py
import numpy as np

class Process:
    def __init__(self, pid, arrival_time, burst_time, priority=1):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.remaining_time = burst_time
        self.waiting_time = 0.0
        self.priority = priority

    def __repr__(self):
        return f"P{self.pid}(Rem:{self.remaining_time:.1f}, Wait:{self.waiting_time:.1f})"


def generate_synthetic_workload(num_processes=50, lambda_arrival=0.5):
    """Generates synthetic CPU burst trace with Poisson process arrivals."""
    processes = []
    current_time = 0.0
    for pid in range(1, num_processes + 1):
        arrival_time = current_time + np.random.exponential(1.0 / lambda_arrival)
        burst_time = float(np.random.randint(5, 50))
        priority = np.random.randint(1, 5)
        processes.append(Process(pid, arrival_time, burst_time, priority))
        current_time = arrival_time
    return processes