import gym
from stable_baselines3 import A2C
import shadow_gym
from time import sleep

# Initialize the environment
env = gym.make("ShadowEnv-v0")

# Load the pre-trained model
model = A2C.load("a2c_hand")

# Reset the environment to start
obs = env.reset()

# Run the simulation
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)
    
    # Optionally render the environment
    # env.render()
    
    # Sleep to simulate real-time (adjust for performance)
    sleep(1/30)
    
    # Check if the episode is done and reset if necessary
    if dones:
        obs = env.reset()

# Close the environment
env.close()