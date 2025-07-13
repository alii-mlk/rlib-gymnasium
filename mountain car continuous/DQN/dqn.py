from ray.rllib.algorithms.dqn import DQNConfig
import ray
import psutil
import time
import pandas as pd
import gymnasium as gym
import torch
import threading
import matplotlib.pyplot as plt
import numpy as np
from ray.tune.registry import register_env
import os
import logging
import warnings

warnings.filterwarnings("ignore")
os.environ["RAY_DEDUP_LOGS"] = "1"
os.environ["RAY_LOG_TO_STDERR"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
logging.getLogger("ray").setLevel(logging.ERROR)

# ----------------------------
# Custom MountainCarContinuous environment with randomized initial state
# ----------------------------
class RandomizedMountainCar(gym.Env):
    def __init__(self, config=None):
        self.env = gym.make("MountainCar-v0")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.spec = gym.envs.registration.EnvSpec("RandomizedMountainCar-v0", max_episode_steps=200)

    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed, options=options)
        position = self.env.np_random.uniform(low=-1.2, high=0.6)
        velocity = self.env.np_random.uniform(low=-0.07, high=0.07)
        self.env.state = np.array([position, velocity], dtype=np.float32)
        return np.array(self.env.state, dtype=np.float32), {}

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


# ----------------------------
# Step 1: Training
# ----------------------------
ray.init(ignore_reinit_error=True)

env_name = "RandomizedMountainCar-v0"
register_env(env_name, lambda config: RandomizedMountainCar())

print(f"\n[INFO] Starting DQN training on {env_name} with RANDOMIZED INITIAL STATES...\n")

process = psutil.Process()
start_time = time.time()
start_mem = process.memory_info().rss / (1024 ** 2)

cpu_log = []
mem_log = []
training_running = True

def monitor_cpu():
    while training_running:
        cpu_log.append(psutil.cpu_percent(interval=0.5))

def monitor_mem():
    while training_running:
        mem_log.append(process.memory_info().rss / (1024 ** 2))
        time.sleep(0.5)

threading.Thread(target=monitor_cpu).start()
threading.Thread(target=monitor_mem).start()

config = (
    DQNConfig()
    .environment(env=env_name, env_config={}, disable_env_checking=True)
    .framework("torch")
    .exploration(explore=True)
    .resources(num_gpus=0)
)

agent = config.build()

target_reward = 100
stable_count = 0
max_stable = 5

for i in range(300  ):
    result = agent.train()
    reward = result["episode_reward_mean"]
    print(f"Iteration {i+1}: episode_reward_mean = {reward:.2f}")
    if reward >= target_reward:
        stable_count += 1
        if stable_count >= max_stable:
            print(f"\n[INFO] Early stopping at iteration {i+1} — reward stabilized ≥ {target_reward} for {max_stable} iterations.\n")
            break
    else:
        stable_count = 0

checkpoint_path = agent.save().checkpoint.path
print(f"\n[INFO] Checkpoint saved at: {checkpoint_path}")

training_running = False
time.sleep(1)

end_time = time.time()
end_mem = process.memory_info().rss / (1024 ** 2)
avg_cpu = sum(cpu_log) / len(cpu_log)
peak_mem = max(mem_log) if mem_log else start_mem
delta_mem = end_mem - start_mem

print("\n[INFO] Training Complete")
print(f"Avg Training CPU Usage: {avg_cpu:.2f}%")
print(f"Peak Memory Usage: {peak_mem:.2f} MB")
print(f"Final Memory Change: {abs(delta_mem):.2f} MB")
print(f"Training Time: {end_time - start_time:.2f} seconds\n")

# ----------------------------
# Step 2: Inference
# ----------------------------
print(f"Running trained DQN policy on {env_name} and logging state evolution...\n")

env = RandomizedMountainCar()
agent = config.build()
agent.restore(checkpoint_path)

obs, _ = env.reset()
done = False
t = 0
log = []

while not done and t < 200:
    cpu = psutil.cpu_percent(interval=0.5)
    ram = process.memory_info().rss / (1024 ** 2)

    action = agent.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    log.append({
        "step": t,
        "cpu": cpu,
        "ram": ram,
        "position": obs[0],
        "velocity": obs[1],
        "reward": reward
    })
    t += 1

df = pd.DataFrame(log)
df.to_csv("mountaincar_inference_DQN_log.csv", index=False)
print("Inference log saved to mountaincar_inference_DQN_log.csv\n")

# ----------------------------
# Step 3: Plot
# ----------------------------
plt.plot(df["step"], df["position"], label="position")
plt.plot(df["step"], df["velocity"], label="velocity")
plt.xlabel("Step")
plt.ylabel("State Value")
plt.title("State Evolution - MountainCar-v0 using DQN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("DQN-MountainCar-v0.png")
plt.show()
print("State plot saved to DQN-MountainCar-v0.png")

ray.shutdown()
