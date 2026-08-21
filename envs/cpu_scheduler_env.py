# RLS/envs/cpu_scheduler_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from config import MAX_QUEUE_SIZE, MAX_BURST_TIME, CONTEXT_SWITCH_COST, ALPHA, BETA, GAMMA
from workload_generator import generate_synthetic_workload

class CPUSchedulerEnv(gym.Env):
    def __init__(self, num_initial_jobs=50):
        super().__init__()
        self.num_initial_jobs = num_initial_jobs
        
        # Action space: Pick the index (0..MAX_QUEUE_SIZE-1) in ready queue to schedule for 1 time unit
        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE)
        
        # State observation vector for each queue slot: [remaining_time, waiting_time, priority]
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1000.0, 
            shape=(MAX_QUEUE_SIZE, 3), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.incoming_jobs = generate_synthetic_workload(self.num_initial_jobs)
        self.ready_queue = []
        self.current_time = 0.0
        self.last_scheduled_pid = None
        self.completed_count = 0
        
        self._update_queue()
        return self._get_obs(), {}

    def _update_queue(self):
        """Pop incoming jobs into ready queue if current_time >= arrival_time."""
        while self.incoming_jobs and self.incoming_jobs[0].arrival_time <= self.current_time:
            if len(self.ready_queue) < MAX_QUEUE_SIZE:
                self.ready_queue.append(self.incoming_jobs.pop(0))
            else:
                break

    def _get_obs(self):
        obs = np.zeros((MAX_QUEUE_SIZE, 3), dtype=np.float32)
        for idx, proc in enumerate(self.ready_queue):
            obs[idx] = [proc.remaining_time, proc.waiting_time, proc.priority]
        return obs

    def step(self, action):
        self._update_queue()
        
        # If queue is empty, advance time
        if not self.ready_queue:
            self.current_time += 1.0
            self._update_queue()
            done = len(self.incoming_jobs) == 0 and len(self.ready_queue) == 0
            return self._get_obs(), 0.0, done, False, {}

        # Clamp action to valid index range
        selected_idx = min(action, len(self.ready_queue) - 1)
        proc = self.ready_queue[selected_idx]
        
        # Check context switch cost
        context_switched = False
        if self.last_scheduled_pid is not None and self.last_scheduled_pid != proc.pid:
            context_switched = True
        self.last_scheduled_pid = proc.pid

        # Execution time step (Time Quantum = 1 time unit)
        execution_step = 1.0
        proc.remaining_time -= execution_step
        self.current_time += execution_step
        
        # Increment waiting time for all OTHER processes sitting in ready queue
        for idx, p in enumerate(self.ready_queue):
            if idx != selected_idx:
                p.waiting_time += execution_step

        # Calculate reward
        avg_wait = np.mean([p.waiting_time for p in self.ready_queue]) if self.ready_queue else 0.0
        cs_penalty = CONTEXT_SWITCH_COST if context_switched else 0.0
        throughput_bonus = 0.0

        # Check process completion
        if proc.remaining_time <= 0:
            self.ready_queue.pop(selected_idx)
            throughput_bonus = GAMMA
            self.completed_count += 1

        reward = -(ALPHA * avg_wait) - (BETA * cs_penalty) + throughput_bonus

        self._update_queue()
        terminated = (len(self.incoming_jobs) == 0 and len(self.ready_queue) == 0)
        
        return self._get_obs(), float(reward), terminated, False, {}