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
# When starting from an existing model, run_name is name of original run and rerun_name is name of logs and models of the new run
start_from_existing = True
existing_model_file = os.path.join("PPO-17c-shadowgym","2500000") # no need .zip extension
re_run_name = "PPO-17c-shadowgym-rerun-2"

# Run name should have model, unique number, and optionally a description
run_name = "PPO-17c-shadowgym"
saving_timesteps_interval = 1000
start_saving = 0

# Set up folders to store models and logs
models_dir = os.path.join(os.path.dirname(__file__), 'models')
logs_dir = os.path.join(os.path.dirname(__file__), 'logs')

if start_from_existing:
    run_dir = os.path.join(models_dir, re_run_name)
    run_logs_dir = os.path.join(logs_dir, re_run_name)
else:
    run_dir = os.path.join(models_dir, run_name)
    run_logs_dir = os.path.join(logs_dir, run_name)

if os.path.exists(run_dir):
    raise Exception("Error: model folder already exists. Change run_name to prevent overriding existing model folder")
if os.path.exists(run_logs_dir):
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
    if os.path.exists(run_dir):
        all_files = os.listdir(run_dir)
        if all_files == []:
            return ""
        file = os.listdir(run_dir)[0]
        return os.path.join(run_dir, file)
    return ""

env = gymnasium.make("ShadowEnv-v0", GUI=False)
env = NormalizeObservation(env)
env = TransformObservation(env,f=clip_observation)
env = Monitor(env)


if start_from_existing:
    previous_model_path = os.path.join(models_dir, existing_model_file)
    model = PPO.load(previous_model_path, env)
else:
    model = PPO(policy="MlpPolicy", env=env, tensorboard_log=logs_dir, normalize_advantage=True, verbose=1, )

timesteps = 0
while True:
    model.learn(saving_timesteps_interval, tb_log_name=run_name, reset_num_timesteps=False)
    timesteps += saving_timesteps_interval
    if timesteps >= start_saving:
        old = get_old_model(run_dir)
        model.save(os.path.join(run_dir, str(timesteps)))
        if old != "":
            os.remove(old)
