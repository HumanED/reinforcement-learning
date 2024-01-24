import pybullet as p
import os
import numpy as np


class Cube:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'cube.urdf')
        startPosition = [0, -0.363, 0.305]
        startOrientation = [np.pi,0,0]
        startOrientationQuaternion = p.getQuaternionFromEuler(startOrientation)
        self.cube = p.loadURDF(f_name,
                               startPosition,  startOrientationQuaternion,
                               physicsClientId=client)
        texture_path = os.path.join(os.path.dirname(__file__), 'cube_texture.jpg')
        x = p.loadTexture(texture_path)
        p.changeVisualShape(self.cube, -1, textureUniqueId=x)

    def get_ids(self):
        return self.cube, self.client

    def apply_action(self, action):
        pass

    def get_observation(self):
        """Returns 12 digit ndarray containing position (x,y,z), orientation in Euler angles (x,y,z), linear velocity (x,y,z) and angular velocity (wx, wy,  wz)"""
        position, orientation = p.getBasePositionAndOrientation(self.cube)
        orientation = p.getEulerFromQuaternion(orientation)
        velocity, angular_velocity = p.getBaseVelocity(self.cube)
        position = np.array(position)
        orientation = np.array(orientation)
        velocity = np.array(velocity)
        angular_velocity = np.array(angular_velocity)

        return np.concatenate((position, orientation, velocity, angular_velocity))
