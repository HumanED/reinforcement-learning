#By Ethan Cheam
#Based on the training_template
import gym
import shadow_gym
from stable_baselines3 import A2C

env = gym.make("ShadowEnv-v0")


model = A2C(policy="MlpPolicy",
            env=env,
            verbose=1)
model.learn(total_timesteps=1_0)
model.save("a2c_hand")

mean_reward, std_reward = evaluate_policy(model, env)
print(f"Mean reward= {mean_reward:.2f} +/- {std_reward:.2f}")  