import gym
from stable_baselines3 import A2C
import shadow_gym
from time import sleep
env = gym.make("ShadowEnv-v0")

model = A2C.load("a2c_hand")
# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)
    #env.render()
    sleep(1/30)

env.close()