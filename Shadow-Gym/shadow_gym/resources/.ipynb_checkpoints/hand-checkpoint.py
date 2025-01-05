import pybullet as p
import numpy as np
import os


class Hand:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__),'shadow_hand.urdf')

        startPosition = [0, 0, 1/4]
        startOrientation = p.getQuaternionFromEuler([np.pi/2, np.pi, 0])
        self.hand_body = p.loadURDF(f_name,
                                    startPosition,
                                    startOrientation,
                                    physicsClientId=client)

    def get_ids(self):
        return self.hand_body, self.client

    def apply_action(self, action):
        joints = [1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30]
        for i in range(len(action)):
            joint_id = joints[i]
            target_position = action[i]
            p.setJointMotorControl2(self.hand_body, joint_id,
                                    p.POSITION_CONTROL,
                                    targetPosition=target_position)
    

