import gym
import shadow_gym
from time import sleep
env = gym.make("ShadowEnv-v0")
observation = env.reset()
for _ in range(1000):
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()

  sleep(1/30)
env.close()