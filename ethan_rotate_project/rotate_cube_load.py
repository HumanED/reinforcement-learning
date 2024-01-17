from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import shadow_gym
import numpy as np
import gym
import os
import time

recurrent = False

# Run name should have model, unique number, and optionally a description
run_name = "PPO" + "-" + "9" + "-" + "shadowgym"
model_file = "2200000.zip"
# Set up folders to store models and logs
models_dir = os.path.join(os.getcwd(),'models')
logs_dir = os.path.join(os.getcwd(),'logs')
model_path = f"{models_dir}/{run_name}/{model_file}"
if not os.path.exists(model_path):
    raise Exception("Error: model not found")

env = gym.make("ShadowEnv-v0")
if recurrent:
    print("Rendering recurrentPPO model...")
    model = RecurrentPPO.load(model_path, env=env)
    num_envs = 1
    lstm_states = None
    episode_starts = np.ones((num_envs,),dtype=bool)
    obs, _ = env.reset()
    while True:
        action, lstm_states = model.predict(obs,state=lstm_states,episode_start=episode_starts, deterministic=True)
        obs, rewards, episode_starts, info = env.step(action)
        time.sleep(1/60)

else:
    print("Running non recurrent PPO model")
    model = PPO.load(model_path,env=env)
    while True:
        done = False
        episode_reward = 0
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            time.sleep(1/60)
        print(f"episode_reward:{episode_reward}")



