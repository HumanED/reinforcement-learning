import gym
import numpy as np
import pybullet as p
from shadow_gym.resources.hand import Hand
from shadow_gym.resources.plane import Plane
import matplotlib.pyplot as plt

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

        self.client = p.connect(p.DIRECT)

        p.setTimeStep(1/30, self.client)
        
        self.hand = None
        self.rendered_img = None

        self.reset()

    def step(self, action):
        self.hand.apply_action(action)
        p.stepSimulation()

        observation = None
        reward = None
        self.done = False

        return observation, reward, self.done, dict()

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        Plane(self.client)
        self.hand = Hand(self.client)

    def render(self):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        hand_id, client_id = self.hand.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(hand_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        p.disconnect(self.client)

    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]