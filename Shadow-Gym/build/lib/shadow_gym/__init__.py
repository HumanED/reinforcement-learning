from gym.envs.registration import register

register(
    id='ShadowEnv-v0', 
    entry_point='shadow_gym.envs:ShadowEnv'
)