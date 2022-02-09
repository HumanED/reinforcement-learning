import gym
import shadow_gym
from stable_baselines3 import PPO

env = gym.make("ShadowEnv-v0")

model = PPO('MlpPolicy', env, learning_rate = 0.001, verbose=1)
model.learn(total_timesteps=500000)
model.save("ppo_hand")