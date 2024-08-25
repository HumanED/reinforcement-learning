import gymnasium
import shadow_gym

env = gymnasium.make("ShadowEnv-v0")
print("obs space",env.observation_space)
obs, info = env.reset()
print("obs", obs)