import gym
import numpy as np
import pybullet as p

wrist_low = np.array([-0.489, -0.785])
wrist_high = np.array([0.140, 0.524])
index_low = np.array([-0.349, 0, 0, 0])
index_high = np.array([0.349, 1.571, 1.571, 1.571])
middle_low = np.array([-0.349, 0, 0, 0])
middle_high = np.array([0.349, 1.571, 1.571, 1.571])
ring_low = np.array([-0.349, 0, 0, 0])
ring_high = np.array([0.349, 1.571, 1.571, 1.571])
little_low = np.array([0, -0.349, 0, 0, 0])
little_high = np.array([0.785, 0.349, 1.571, 1.571, 1.571])
thumb_low = np.array([-0.960, 0, -0.209, -0.436, 0])
thumb_high = np.array([0.960, 1.222, 0.209, 0.436, 1.571])



class ShadowEnv(gym.Env):
    metadata = {'render.modes': ['human']}  
  
    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low = np.concatenate((wrist_low, index_low, middle_low, ring_low, little_low, thumb_low)),
            high = np.concatenate((wrist_high, index_high, middle_high, ring_high, little_high, thumb_high))
        )
        self.observation_space = gym.spaces.box.Box(
            low = np.array([-10, -10, -10]),
            high = np.array([10, 10, 10])
        )
        self.np_random, _ = gym.utils.seeding.np_random()
        

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass    
    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]