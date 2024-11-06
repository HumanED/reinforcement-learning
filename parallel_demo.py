import os
import time
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium
import numpy as np
import shadow_gym  

# Settings
recurrent = False  
vectorized_env = False
normalized_env = False  
num_evaluate = -1  # Set to -1 to run in GUI mode until windows closed
run_name = "PPO-19-shadowgym-ethan"
model_file = "7900000.zip"
normalize_stats_file = "1800000.pkl" 

# Model and log directories
models_dir = os.path.join(os.path.dirname(__file__), 'models')
logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
normalize_stats_path = os.path.join(os.path.dirname(__file__), 'normalize_stats', run_name, normalize_stats_file)
model_path = os.path.join(models_dir, run_name, model_file)

# Ensure model exists
if not os.path.exists(model_path):
    raise Exception("Error: model not found")

# Function to run a single instance of the environment visualization
def visualize_instance(instance_id, model_path, normalize_stats_file=None, GUI=False):
    # Initialize the environment with or without normalization
    if vectorized_env:
        env = DummyVecEnv([lambda: gymnasium.make("ShadowEnv-v0", GUI=GUI)])
        if normalized_env and normalize_stats_file:
            env = VecNormalize.load(normalize_stats_file, env)
            env.training = False
            env.norm_reward = False
    else:
        env = gymnasium.make("ShadowEnv-v0", GUI=GUI)

    # Load the model
    print(f"Running instance {instance_id} - PPO model")
    if recurrent:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path, env=env)
        num_envs = 1
        lstm_states = None
        episode_starts = np.ones((num_envs,), dtype=bool)
        obs, _ = env.reset()
        while True:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, rewards, episode_starts, info = env.step(action)
            time.sleep(1 / 60)
    else:
        model = PPO.load(model_path, env=env)
        total_success = 0
        episode_count = 0
        run_forever = GUI

        # Run episodes
        while episode_count < num_evaluate or run_forever:
            terminated = False
            truncated = False
            episode_reward = 0
            obs, _ = env.reset()
            while not terminated and not truncated:
                action, _ = model.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                time.sleep(1 / 24)  
            print(f"Instance {instance_id} - Episode Reward: {episode_reward}")
            if info.get("success", False):
                total_success += 1
            episode_count += 1
            print(f"Instance {instance_id} - Total Success: {total_success}, Episodes: {episode_count}, "
                  f"Success Rate: {total_success / episode_count:.2f}")

        env.close()

# Main function to run multiple instances
def run_instance(instance_id):
    GUI = True if num_evaluate == -1 else False 
    visualize_instance(
        instance_id=instance_id,
        model_path=model_path,
        normalize_stats_file=normalize_stats_path if normalized_env else None,
        GUI=GUI
    )


if __name__ == "__main__":
    # list to store the instances
    processes = []
    for i in range(3):  # Number of instances we want
        #create the process instance and set the function to be run 
        p = multiprocessing.Process(target=run_instance, args=(i,))
        #start the instance
        p.start()
        #add the running instances to a list 
        processes.append(p)

    # terminate only when all windows are closed
    for p in processes:
        # stop the program from terminating
        p.join()
