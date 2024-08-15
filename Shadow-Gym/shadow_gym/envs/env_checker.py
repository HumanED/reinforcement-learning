from stable_baselines3.common.env_checker import check_env
from shadow_gym.envs.shadow_env import ShadowEnv
# Ignore the results of the box system.
env = ShadowEnv()
check_env(env)
print("All ok")