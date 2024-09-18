import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import gymnasium
from gymnasium.wrappers.normalize import NormalizeObservation
from gymnasium.wrappers.transform_observation import TransformObservation
import numpy as np
import shadow_gym

"""
Author: Ethan Cheam
Trains a PPO model, saves models at regular intervals, and record training performance in Tensorboard.
"""

# SETTINGS
start_from_existing = True
existing_model_file = "2000000" # no need .zip extension

# Run name should have model, unique number, and optionally a description
# 17c is a long run of a hopefully complete model with clipping and norm obs
run_name = "PPO" + "-" + "17c" + "-" + "shadowgym"
saving_timesteps_interval = 100_000 # Reduce this to 100_000 instead
start_saving = 500_000

# Set up folders to store models and logs
models_dir = os.path.join(os.path.dirname(__file__), 'models')
if not os.path.exists(models_dir):
    os.mkdir(models_dir)
logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
run_dir = os.path.join(models_dir,run_name)
if not start_from_existing and os.path.exists(run_dir):
    raise Exception("Error: model folder already exists. Change run_name to prevent overriding existing model folder")
else:
    temp_run_dir = os.path.join(models_dir, run_name + "_existing_model")
    os.mkdir(temp_run_dir)
if not start_from_existing and os.path.exists(f"{logs_dir}/{run_name}"):
    raise Exception("Error: log folder already exists. Change run_name to prevent overriding existing log folder")

def clip_observation(obs):
    """
    clips observation to within 5 standard deviations of the mean
    Refer to section D.1 of Open AI paper
    """
    return np.clip(obs,a_min=obs.mean() - (5 * obs.std()), a_max=obs.mean() + (5 * obs.std()))

def get_old_model(run_dir) -> str:
    """
    Returns full path of old model to delete. A model must end in .zip.
    Before first model is saved, returns "" indicating no existing model
    """
    all_files = os.listdir(run_dir)
    if all_files == []:
        return ""
    file = os.listdir(run_dir)[0]
    return os.path.join(run_dir, file)

env = gymnasium.make("ShadowEnv-v0", GUI=False)
env = NormalizeObservation(env)
env = TransformObservation(env,f=clip_observation)
env = Monitor(env)


if start_from_existing:
    full_model_path = os.path.join(models_dir, run_name, existing_model_file)
    model = PPO.load(full_model_path, env)
else:
    model = PPO(policy="MlpPolicy", env=env, tensorboard_log=logs_dir, normalize_advantage=True, verbose=1, )

timesteps = 0
while True:
    model.learn(saving_timesteps_interval, tb_log_name=run_name, reset_num_timesteps=False)
    timesteps += saving_timesteps_interval
    if timesteps >= start_saving:
        old = get_old_model(run_dir)
        model.save(f"{models_dir}/{run_name}/{timesteps}")
        if old != "":
            os.remove(old)
