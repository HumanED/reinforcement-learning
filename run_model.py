import gym
import shadow_gym
from stable_baselines3 import A2C
from time import sleep

env = gym.make("ShadowEnv-v0")

model = A2C.load("ppo_hand")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()
    sleep(1/30)
