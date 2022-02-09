import gym
import shadow_gym
from stable_baselines3 import A2C

env = gym.make("ShadowEnv-v0")

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
model.save("a2c_hand")