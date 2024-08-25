import os
import time
from statistics import mean, stdev
from stable_baselines3 import PPO
from gymnasium.wrappers.normalize import NormalizeObservation
import numpy as np
import gymnasium
import shadow_gym
from tqdm.auto import tqdm

# SETTINGS
num_ep_evaluate = 10
# Run name should have model, unique number, and optionally a description
model_folder_zip = "PPO-17b-shadowgym/30000.zip"


# Set up folders to store models and logs
model_path = os.path.join(os.path.dirname(__file__),"models",model_folder_zip)
if not os.path.exists(model_path):
    raise Exception("Error: model not found")

env = gymnasium.make("ShadowEnv-v0", GUI=False)
env = NormalizeObservation(env)


print(f"Evaluating non recurrent PPO model {model_folder_zip}")
model = PPO.load(model_path,env=env)
total_success = 0
episode_count = 0
episode_info = {}
episode_rewards = []
obs, info = env.reset()
for key in info.keys():
    episode_info[key] = []
for episode in tqdm(range(num_ep_evaluate)):
    terminated = False
    truncated = False
    obs, info = env.reset()
    episode_reward = 0
    while not terminated and not truncated:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
    for key in info.keys():
        episode_info[key].append(info[key])
    episode_rewards.append(episode_reward)

print(f"episode_rewards",episode_rewards)
for key, value in episode_info.items():
    print(key, value)
print("-------------------------------------")
print(f"episode_rewards          mean: {mean(episode_rewards)} std: {stdev(episode_rewards)} ")
for key in episode_info.keys():
    print(f"{key:20} mean: {mean(episode_info[key])} std: {stdev(episode_info[key])}")
env.close()



