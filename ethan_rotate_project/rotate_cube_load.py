from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import shadow_gym
import numpy as np
import gym
import os
import time

recurrent = False
num_evaluate = -1 # Set num_evaluate to -1 to enable rendering and just view the project

# Run name should have model, unique number, and optionally a description
run_name = "PPO" + "-" + "12" + "-" + "shadowgym"
model_file = "2800000.zip"
# Set up folders to store models and logs
models_dir = os.path.join(os.getcwd(),'models')
logs_dir = os.path.join(os.getcwd(),'logs')
model_path = f"{models_dir}/{run_name}/{model_file}"
if not os.path.exists(model_path):
    raise Exception("Error: model not found")

GUI=False
if num_evaluate == -1:
    GUI = True
env = gym.make("ShadowEnv-v0",GUI=GUI)
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
    total_success = 0
    episode_count = 0
    if num_evaluate == -1:
        run_forever=True
    else:
        run_forever=False
    while episode_count < num_evaluate or run_forever:
        done = False
        episode_reward = 0
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            # time.sleep(1/60)
        print(f"episode_reward:{episode_reward}")
        if info["success"]:
            total_success += 1
        episode_count += 1
        print(f"total_success:{total_success} episodes:{episode_count} ratio:{total_success / episode_count}")




