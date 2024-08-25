import gymnasium
import shadow_gym
from time import sleep
env = gymnasium.make("ShadowEnv-v0",GUI=True)
observation, _ = env.reset()
terminated = False
terminated = False
for _ in range(1000):
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, terminated, truncated , info = env.step(action)

  if terminated or truncated:
    observation = env.reset()


  sleep(1/30)
env.close()