import pybullet as p
import os
import numpy as np
from pyquaternion import Quaternion

class Cube:
    def __init__(self, client, start_orientation_q):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'cube.urdf')
        start_position = [0, -0.363, 0.305]
        self.cube_body = p.loadURDF(f_name,
                                    start_position, start_orientation_q,
                                    physicsClientId=client)
        texture_path = os.path.join(os.path.dirname(__file__), 'cube_texture.jpg')
        x = p.loadTexture(texture_path)
        p.changeVisualShape(self.cube_body, -1, textureUniqueId=x)

    def get_ids(self):
        return self.cube_body, self.client

    def apply_action(self, action):
        pass


