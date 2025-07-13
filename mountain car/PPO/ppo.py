import ray
import psutil
import time
import pandas as pd
import gymnasium as gym
import torch
import threading
import matplotlib.pyplot as plt
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from gym.envs.classic_control.mountain_car import MountainCarEnv

# ----------------------------
# Custom MountainCar environment with randomized initial state
# ----------------------------
class RandomizedMountainCar(MountainCarEnv):
    def __init__(self):
        super().__init__()
        self.spec = gym.envs.registration.EnvSpec("RandomizedMountainCar-v1", max_episode_steps=200)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        position = self.np_random.uniform(low=-1.2, high=0.6)    
        velocity = self.np_random.uniform(low=-0.07, high=0.07)   
        self.state = np.array([position, velocity], dtype=np.float32)
        return np.array(self.state, dtype=np.float32)

# ----------------------------
# Step 1: Training
# ----------------------------
ray.init(ignore_reinit_error=True)

env_name = "RandomizedMountainCar-v1"
register_env(env_name, lambda config: RandomizedMountainCar())

print(f"\n[INFO] Starting PPO training on {env_name} with RANDOMIZED INITIAL STATES...\n")

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

cpu_thread = threading.Thread(target=monitor_cpu)
mem_thread = threading.Thread(target=monitor_mem)
cpu_thread.start()
mem_thread.start()

config = (
    PPOConfig()
    .environment(env=env_name, env_config={}, disable_env_checking=True)
    .framework("torch")
    .rollouts(horizon=200)
    .exploration(explore=True)
)

agent = config.build()

target_reward = -90
stable_count = 0
max_stable = 5

for i in range(300):
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

checkpoint_path = agent.save()
print(f"\n[INFO] Checkpoint saved at: {checkpoint_path}")

training_running = False
cpu_thread.join()
mem_thread.join()

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
print(f"Running trained PPO policy on {env_name} and logging state evolution...\n")

env = RandomizedMountainCar()
agent = config.build()
agent.restore(checkpoint_path)

obs = env.reset()
done = False
t = 0
log = []

while not done and t < 200:
    cpu = psutil.cpu_percent(interval=0.5)
    ram = process.memory_info().rss / (1024 ** 2)

    action = agent.compute_single_action(obs)
    obs, reward, done, info = env.step(action)

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
df.to_csv("mountaincar_inference_PPO_log.csv", index=False)
print("Inference log saved to mountaincar_inference_PPO_log.csv\n")

# ----------------------------
# Step 3: Plot
# ----------------------------
plt.plot(df["step"], df["position"], label="position")
plt.plot(df["step"], df["velocity"], label="velocity")
plt.xlabel("Step")
plt.ylabel("State Value")
plt.title("State Evolution - MountainCar-v1 using PPO")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("PPO-MountainCar-v1.png")
plt.show()
print("State plot saved to PPO-MountainCar-v1.png")

ray.shutdown()


