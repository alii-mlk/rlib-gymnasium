import ray
import psutil
import time
import pandas as pd
import gymnasium as gym
import torch
import threading
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
from gym.envs.classic_control.cartpole import CartPoleEnv

# ----------------------------
# Custom CartPole environment with randomized initial state
# ----------------------------
class RandomizedCartPole(CartPoleEnv):
    def __init__(self):
        super().__init__()
        self.spec = gym.envs.registration.EnvSpec("RandomizedCartPole-v1", max_episode_steps=500)
        self.observation_space = Box(
            low=np.array([-4.8, -5.0, -0.418, -5.0], dtype=np.float32),
            high=np.array([4.8, 5.0, 0.418, 5.0], dtype=np.float32),
            dtype=np.float32
        )
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(
            low=np.array([-0.25, -1.0, -0.15, -1.0]),
            high=np.array([0.25, 1.0, 0.15, 1.0])
        )
        return np.array(self.state, dtype=np.float32)

# ----------------------------
# Step 1: Training
# ----------------------------
ray.init(ignore_reinit_error=True)

env_name = "RandomizedCartPole-v1"
register_env(env_name, lambda config: RandomizedCartPole())

print(f"\nStarting DQN training on {env_name} with RANDOMIZED INITIAL STATES...\n")

process = psutil.Process()
start_time = time.time()
start_mem = process.memory_info().rss / (1024 ** 2)

cpu_log = []
mem_log = []
training_running = True

def monitor_cpu():
    while training_running:
        cpu_log.append(psutil.cpu_percent(interval=1))

def monitor_mem():
    while training_running:
        mem_log.append(process.memory_info().rss / (1024 ** 2))
        time.sleep(0.5)

cpu_thread = threading.Thread(target=monitor_cpu)
mem_thread = threading.Thread(target=monitor_mem)
cpu_thread.start()
mem_thread.start()

config = (
    DQNConfig()
    .environment(env=env_name, env_config={}, disable_env_checking=True)
    .framework("torch")
    .rollouts(horizon=500)
)

agent = config.build()

target_reward = 500
stable_count = 0
max_stable = 5

for i in range(300):
    result = agent.train()
    reward = result["episode_reward_mean"]
    print(f"Iteration {i+1}: episode_reward_mean = {reward:.2f}")
    if reward >= target_reward:
        stable_count += 1
        if stable_count >= max_stable:
            print(f"\nEarly stopping at iteration {i+1} — reward stabilized at 500 for {max_stable} iterations.\n")
            break
    else:
        stable_count = 0

checkpoint_path = agent.save()
print(f"\nCheckpoint saved at: {checkpoint_path}")

training_running = False
cpu_thread.join()
mem_thread.join()

end_time = time.time()
end_mem = process.memory_info().rss / (1024 ** 2)
avg_cpu = sum(cpu_log) / len(cpu_log)
peak_mem = max(mem_log) if mem_log else start_mem
delta_mem = end_mem - start_mem

print("\nTraining Complete")
print(f"Avg Training CPU Usage: {avg_cpu:.2f}%")
print(f"Peak Memory Usage: {peak_mem:.2f} MB")
print(f"Final Memory Change: {delta_mem:.2f} MB")
print(f"Training Time: {end_time - start_time:.2f} seconds\n")

# ----------------------------
# Step 2: Inference
# ----------------------------
print(f"Running trained DQN policy on {env_name} and logging state evolution...\n")

env = RandomizedCartPole()
agent = config.build()
agent.restore(checkpoint_path)

obs = env.reset()
done = False
t = 0
log = []

while not done and t < 500:
    cpu = psutil.cpu_percent(interval=0.5)
    ram = process.memory_info().rss / (1024 ** 2)

    action = agent.compute_single_action(obs)
    obs, reward, done, info = env.step(action)

    log.append({
        "step": t,
        "cpu": cpu,
        "ram": ram,
        "state_cart_pos": obs[0],
        "state_cart_vel": obs[1],
        "state_pole_angle": obs[2],
        "state_pole_vel": obs[3],
        "reward": reward
    })
    t += 1

df = pd.DataFrame(log)
df.to_csv("cartpole_inference_DQN_log.csv", index=False)
print("Inference log saved to cartpole_inference_DQN_log.csv\n")

# ----------------------------
# Step 3: Plot
# ----------------------------
plt.plot(df["step"], df["state_cart_pos"], label="Cart Position")
plt.plot(df["step"], df["state_pole_angle"], label="Pole Angle")
plt.xlabel("Step")
plt.ylabel("State Value")
plt.title("State Evolution - CartPole-v1 using DQN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("DQN-CartPole-v1.png")
plt.show()
print("State plot saved to DQN-CartPole-v1.png")

ray.shutdown()
