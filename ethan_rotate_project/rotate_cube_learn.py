from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import os
import shadow_gym
import gym

"""
Created by Ethan Cheam
Trains a PPO model, saves models at defined intervals and record training performance in Tensorboard.
"""

# SETTINGS
# RecurrentPPO or PPO
recurrent = False
vectorized_env = True
normalized_env = True
# Run name should have model, unique number, and optionally a description
run_name = "PPO" + "-" + "14" + "-" + "shadowgym"
saving_timesteps_interval = 1000
start_saving = 0

# Set up folders to store models and logs
models_dir = os.path.join(os.path.dirname(__file__), 'models')
logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
normalize_stats = os.path.join(os.path.dirname(__file__), 'normalize_stats')
if os.path.exists(f"{models_dir}/{run_name}"):
    raise Exception("Error: model folder already exists. Change run_name to prevent overriding existing model folder")
if os.path.exists(f"{logs_dir}/{run_name}"):
    raise Exception("Error: log folder already exists. Change run_name to prevent overriding existing log folder")
if os.path.exists(f"{normalize_stats}/{run_name}"):
    raise Exception("Error: normalize_stats folder already exists. Change run_name")
os.makedirs(f"{normalize_stats}/{run_name}")


if vectorized_env:
    env = DummyVecEnv([lambda: gym.make("ShadowEnv-v0", GUI=False)])
    if normalized_env:
        env = VecNormalize(env)
else:
    env = gym.make("ShadowEnv-v0", GUI=False)

if recurrent:
    from sb3_contrib import RecurrentPPO

    model = RecurrentPPO(policy="MlpLstmPolicy", env=env, tensorboard_log=logs_dir, verbose=1)
else:
    model = PPO(policy="MlpPolicy", env=env, tensorboard_log=logs_dir, verbose=0)

timesteps = 0
while True:
    model.learn(saving_timesteps_interval, tb_log_name=run_name, reset_num_timesteps=False)
    timesteps += saving_timesteps_interval
    if timesteps >= start_saving:
        model.save(f"{models_dir}/{run_name}/{timesteps}")
        if vectorized_env and normalized_env:
            normalize_stats_path = os.path.join(normalize_stats, run_name, str(timesteps) + '.pkl')
            env.save(normalize_stats_path)
