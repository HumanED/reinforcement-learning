import pybullet as p
import numpy as np
import os


class Hand:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__),
                              'shadow_hand.urdf')

        self.startPosition = [0, 0, 1/4]
        self.startOrientation = p.getQuaternionFromEuler([np.pi/2, np.pi, 0])
        self.hand = p.loadURDF(f_name,
                               self.startPosition,
                               self.startOrientation,
                               physicsClientId=client)

        self.joints = [1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30]
    def get_ids(self):
        return self.hand, self.client

    def apply_action(self, action):
        for i in range(len(action)):
            joint_id = self.joints[i]
            target_position = action[i]
            p.setJointMotorControl2(self.hand, joint_id,
                                    p.POSITION_CONTROL,
                                    targetPosition=target_position)
    
    def get_observation(self):
        joint_position = []
        for joint_id in self.joints:
            joint_position.append(p.getJointState(self.hand, joint_id)[0])

        return np.array(joint_position)
        