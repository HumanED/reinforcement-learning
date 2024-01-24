from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import os
import shadow_gym
import gym
"""
Created by Ethan Cheam
Trains a PPO model, saves models at defined intervals and record training performance in Tensorboard.
"""

# RecurrentPPO or PPO
recurrent = False
# Run name should have model, unique number, and optionally a description
# 9 is the first discretization (NOTEME)
run_name = "PPO" + "-" + "13" + "-" + "shadowgym"
# Set up folders to store models and logs
models_dir = os.path.join(os.path.dirname(__file__),'models')
logs_dir = os.path.join(os.path.dirname(__file__),'logs')
if os.path.exists(f"{models_dir}/{run_name}"):
    raise Exception("Error: model folder already exists. Change run_name to prevent overriding existing model folder")
if os.path.exists(f"{logs_dir}/{run_name}"):
    raise Exception("Error: log folder already exists. Change run_name to prevent overriding existing log folder")

env = gym.make("ShadowEnv-v0",GUI=False)
if recurrent:
    model = RecurrentPPO(policy="MlpLstmPolicy",env=env,tensorboard_log=logs_dir,verbose=1)
else:
    model = PPO(policy="MlpPolicy",env=env,tensorboard_log=logs_dir,verbose=0)

saving_timesteps_interval = 50_000
start_saving = 500_000
timesteps = 0
while True:
    model.learn(saving_timesteps_interval, tb_log_name=run_name, reset_num_timesteps=False)
    timesteps += saving_timesteps_interval
    if timesteps >= start_saving:
        model.save(f"{models_dir}/{run_name}/{timesteps}")