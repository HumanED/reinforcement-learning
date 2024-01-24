from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import gym
import shadow_gym
"""
NOT YET FUNCTIONING 24/1/2024
"""

run_name = "PPO" + "-" + "6" + "-" + "shadowgym"
run_dir = os.path.join(os.getcwd(), 'models', run_name)
if os.path.exists(run_dir):
    raise Exception("Error: run already exists")

env_id = "ShadowEnv-v0"
num_training = 1
num_evaluation = 5
training_env = DummyVecEnv([lambda: gym.make(env_id)])


def outer(env_id, rank):
    def inner():
        env = gym.make(env_id)
        env.reset(seed=rank)
        return env

    return inner


eval_env = DummyVecEnv([outer(env_id, 0) for _ in range(num_evaluation)])
callback = EvalCallback(eval_env, best_model_save_path=run_dir, eval_freq=(1000 // num_training), log_path=run_dir,
                        deterministic=True, render=False, verbose=0)

logs_dir = os.path.join(os.getcwd(), 'logs')
if os.path.exists(f"{logs_dir}/{run_name}"):
    raise Exception("Error: log folder already exists. Change run_name to prevent overriding existing log folder")

model = PPO(policy="MlpPolicy", env=training_env, tensorboard_log=logs_dir)
model.learn(2_000_000, callback=callback, tb_log_name=run_name)
