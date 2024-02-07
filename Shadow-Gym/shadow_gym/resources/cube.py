import pybullet as p
import os
import numpy as np


class Cube:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'cube.urdf')
        startPosition = [0, -0.363, 0.305]
        startOrientation = [np.random.randint(0, 3) * (np.pi / 2),
                            np.random.randint(0, 3) * (np.pi / 2),
                            np.random.randint(0, 3) * (np.pi / 2)]
        startOrientationQuaternion = p.getQuaternionFromEuler(startOrientation)
        self.cube = p.loadURDF(f_name,
                               startPosition, startOrientationQuaternion,
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
        position, orientation_quaternion = p.getBasePositionAndOrientation(self.cube)
        velocity, angular_velocity = p.getBaseVelocity(self.cube)
        position = np.array(position)
        orientation_quaternion = np.array(orientation_quaternion)
        velocity = np.array(velocity)
        angular_velocity = np.array(angular_velocity)

        return np.concatenate((position, orientation_quaternion, velocity, angular_velocity))
