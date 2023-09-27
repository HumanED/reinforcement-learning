import pybullet as p
import os
import numpy as np



class Cube:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__),
                              'cube.urdf')
        # Default start position.
        # startPosition = [0, -1/3.5, 1/3]
        # Start position over the fingers. Same height as default
        startPosition = [0, -1/3.5 - 0.08, 1/3]
        # startPosition tester
        # startPosition = [0, -1/3.5, 1/3]
        
        self.cube = p.loadURDF(f_name,
                               startPosition,
                               physicsClientId=client)

    def get_ids(self):
        return self.cube, self.client

    def apply_action(self, action):
        pass
    
    def get_observation(self):
        # Modified cube to include velocity
        position, orientation = p.getBasePositionAndOrientation(self.cube)
        orientation = p.getEulerFromQuaternion(orientation)
        velocityXYZ = p.getBaseVelocity(self.cube)[0]
        
        position = np.array(position)
        orientation = np.array(orientation)
        velocityXYZ = np.array(velocityXYZ)

        return np.concatenate((position, orientation, velocityXYZ))