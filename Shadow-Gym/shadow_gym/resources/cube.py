import pybullet as p
import os
import numpy as np



class Cube:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__),
                              'cube.urdf')

        startPosition = [0, -1/3.5, 1/3]
        self.cube = p.loadURDF(f_name,
                               startPosition,
                               physicsClientId=client)

    def get_ids(self):
        return self.cube, self.client

    def apply_action(self, action):
        pass
    
    def get_observation(self):
        position, orientation = p.getBasePositionAndOrientation(self.cube)
        orientation = p.getEulerFromQuaternion(orientation)
        position = np.array(position)
        orientation = np.array(orientation)

        return np.concatenate((position, orientation))