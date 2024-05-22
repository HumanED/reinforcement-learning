from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

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
vectorized_env = False
normalized_env = False
start_from_existing = False
existing_model_file = "" # no need .zip extension

# Run name should have model, unique number, and optionally a description
run_name = "PPO" + "-" + "16" + "-" + "shadowgym"
saving_timesteps_interval = 50_000
start_saving = 1_000_000

# Set up folders to store models and logs
models_dir = os.path.join(os.path.dirname(__file__), 'models')
logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
normalize_stats = os.path.join(os.path.dirname(__file__), 'normalize_stats')
if not start_from_existing and os.path.exists(f"{models_dir}/{run_name}"):
    raise Exception("Error: model folder already exists. Change run_name to prevent overriding existing model folder")
if not start_from_existing and os.path.exists(f"{logs_dir}/{run_name}"):
    raise Exception("Error: log folder already exists. Change run_name to prevent overriding existing log folder")
if not start_from_existing and os.path.exists(f"{normalize_stats}/{run_name}"):
    raise Exception("Error: normalize_stats folder already exists. Change run_name")
if normalized_env:
    os.makedirs(f"{normalize_stats}/{run_name}")

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_length = 0
        self.episode_reward = 0
    def _on_rollout_end(self) -> None:
        # Error: _on_rollout and episodes are different time periods.
        self.episode_length = self.training_env.get_attr("num_steps")[0]
        self.logger.record("rollout/ep_len_mean",self.episode_length)
        self.logger.record("rollout/ep_rew_mean", self.episode_reward / self.episode_length)
        # Reset vars
        self.episode_length = 0
        self.episode_reward = 0

    def _on_step(self) -> bool:
        self.episode_reward += self.training_env.get_attr("reward")[0]
        return True

rewards_callback = None
if vectorized_env:
    env = DummyVecEnv([lambda: gym.make("ShadowEnv-v0", GUI=False)])
    if normalized_env:
        env = VecNormalize(env)
    rewards_callback = TensorboardCallback()
else:
    env = gym.make("ShadowEnv-v0", GUI=False)

full_model_path = None
if start_from_existing:
    full_model_path = os.path.join(models_dir, run_name, existing_model_file)
if recurrent:
    from sb3_contrib import RecurrentPPO
    if start_from_existing:
        model = RecurrentPPO.load(full_model_path, env)
    else:
        model = RecurrentPPO(policy="MlpLstmPolicy", env=env, tensorboard_log=logs_dir, verbose=1)
else:
    if start_from_existing:
        model = PPO.load(full_model_path, env)
    else:
        model = PPO(policy="MlpPolicy", env=env, tensorboard_log=logs_dir, verbose=1)

timesteps = 0
while True:
    model.learn(1000, tb_log_name=run_name, reset_num_timesteps=False)
    timesteps += saving_timesteps_interval
    if timesteps >= start_saving:
        model.save(f"{models_dir}/{run_name}/{timesteps}")
        if vectorized_env and normalized_env:
            normalize_stats_path = os.path.join(normalize_stats, run_name, str(timesteps) + '.pkl')
            env.save(normalize_stats_path)
