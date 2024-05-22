import gym
import numpy as np
import pybullet as p
from shadow_gym.resources.hand import Hand
from shadow_gym.resources.plane import Plane
from shadow_gym.resources.cube import Cube
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
import random

# Changed by Ethan
discretize = True
number_of_bins = 11

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

if discretize:
    wrist_bin_size = (wrist_high - wrist_low) / number_of_bins
    index_bin_size = (index_high - index_low) / number_of_bins
    middle_bin_size = (middle_high - middle_low) / number_of_bins
    ring_bin_size = (ring_high - ring_low) / number_of_bins
    little_bin_size = (little_high - little_low) / number_of_bins
    thumb_bin_size = (thumb_high - thumb_low) / number_of_bins
    bin_sizes = np.concatenate(
        (wrist_bin_size, index_bin_size, middle_bin_size, ring_bin_size, little_bin_size, thumb_bin_size,))

hand_motion_low = np.concatenate((wrist_low, index_low, middle_low, ring_low, little_low, thumb_low))
hand_motion_high = np.concatenate((wrist_high, index_high, middle_high, ring_high, little_high, thumb_high))

hand_velocity_high = np.array([np.inf] * 96)
hand_velocity_low = np.array([-np.inf] * 96)


def calculate_angular_difference(orientation1, orientation2):
    """
    Calculates angular difference in radians between two quaternions
    :param orientation1: List[int]
    :param orientation2: List[int]
    """
    rot1 = Rotation.from_quat(orientation1)
    rot2 = Rotation.from_quat(orientation2)
    # .inv() is transform of the rotation matrix
    angular_difference = rot1.inv() * rot2
    #  axis-angle representation of angular difference
    axis_angle = angular_difference.as_rotvec()
    # Convert angular difference into axis-angle representation
    rotation_magnitude = np.linalg.norm(axis_angle)
    return rotation_magnitude


class ShadowEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, GUI=False):
        """
        :param GUI: `GUI=True` after training. `GUI=False` for during training
        """
        self.reward = None
        if discretize:
            # 11 bins for each of the 24 actions
            self.action_space = gym.spaces.MultiDiscrete([11] * 24)
        else:
            self.action_space = gym.spaces.box.Box(
                low=hand_motion_low,
                high=hand_motion_high
            )

        self.observation_space = gym.spaces.box.Box(
            # Cube observation is 7 digit ndarray with position (x,y,z) and relative quaternion (w,x,y,z),
            # orientation in quaternion (a,b,c,d), linear velocity (x,y,z) and angular velocity (wx, wy,  wz)
            low=np.concatenate((hand_motion_low, hand_velocity_low, np.array(
                [-np.inf, -np.inf, -np.inf, -1, -1, -1, -1]))),
            high=np.concatenate((hand_motion_high, hand_velocity_high, np.array(
                [np.inf, np.inf, np.inf, 1, 1, 1, 1])))
        )

        self.np_random, _ = gym.utils.seeding.np_random()

        if GUI:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setTimeStep(1 / 30, self.client)

        self.hand = None
        self.cube = None
        self.rendered_img = None
        self.done = False
        self.num_steps = 0

        self.previous_rotation_to_target = 4
        self.target_euler = [0, 0, 0]
        self.target_quaternion = p.getQuaternionFromEuler(self.target_euler)
        self.STEP_LIMIT = 300  # Given a timestep is 1/30 seconds.

        self.episodes_before_change = 100
        self.episode_counter = 0
        self.reset()

    def get_cube_observation(self, target_orientation_q: list[int]):
        """Returns 7 digit ndarray containing position (x,y,z), relative orientation quaternion (w, x, y, z)"""
        position, orientation_quaternion = p.getBasePositionAndOrientation(self.cube.cube_body)
        q1 = Quaternion(target_orientation_q)
        q2 = Quaternion(orientation_quaternion)
        relative_q = q1 * q2.inverse
        relative_q = list(relative_q)
        return np.concatenate((position, relative_q))

    def get_hand_observation(self):
        joints = [1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30]
        joint_position = []

        # Adding the velocities (x,y,z) of each link
        num_links = p.getNumJoints(self.hand.hand_body)
        link_velocity = []
        for i in range(num_links):
            link_velocity.extend(p.getLinkState(self.hand.hand_body, i, computeLinkVelocity=True)[7])

        for joint_id in joints:
            joint_position.append(p.getJointState(self.hand.hand_body, joint_id)[0])

        return np.array(joint_position + link_velocity)

    def step(self, action):
        self.num_steps += 1
        if discretize:
            # Convert discrete action choice from the AI to a continuous action for the motor.
            action = hand_motion_low + (bin_sizes / 2) + (bin_sizes * action)

        self.hand.apply_action(action)
        p.stepSimulation()

        hand_observation = self.get_hand_observation()
        cube_observation = self.get_cube_observation(self.target_quaternion)

        observation = np.concatenate((hand_observation, cube_observation))

        info = {"success": False}
        # Reward calculations
        cube_orientation_q = p.getBasePositionAndOrientation(self.cube.cube_body)[1]
        rotation_to_target = calculate_angular_difference(self.target_quaternion, cube_orientation_q)
        self.reward = self.previous_rotation_to_target - rotation_to_target

        if rotation_to_target < 0.4:
            # We are less than 0.4 radians (23 degrees to target)
            self.reward = 5
            self.done = True
            info["success"] = True
        if cube_observation[2] < 0.05:
            # Cube is dropped
            self.reward = -20
            self.done = True
        if self.num_steps > self.STEP_LIMIT:  # 300
            self.done = True
        self.previous_rotation_to_target = rotation_to_target

        return observation, self.reward, self.done, info

    def reset(self):
        if self.episode_counter == 0:
            self.episode_counter = self.episodes_before_change
            euler = [np.random.randint(0, 3) * (np.pi / 2),
                     np.random.randint(0, 3) * (np.pi / 2),
                     np.random.randint(0, 3) * (np.pi / 2)]
            while euler != self.target_euler:
                euler = [np.random.randint(0, 3) * (np.pi / 2),
                         np.random.randint(0, 3) * (np.pi / 2),
                         np.random.randint(0, 3) * (np.pi / 2)]
            self.cube_start_orientation = euler
        self.episode_counter -= 1

        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        Plane(self.client)
        self.hand = Hand(self.client)
        random_orientation_euler = (random.randint(-1,1) * np.pi/2,
                                    random.randint(-1,1) * np.pi/2,
                                    random.randint(-1,1) * np.pi/2)
        random_orientation_q = p.getQuaternionFromEuler(random_orientation_euler)
        self.cube = Cube(self.client, random_orientation_q)

        self.done = False
        self.num_steps = 0

        hand_observation = self.get_hand_observation()
        cube_observation = self.get_cube_observation(self.target_quaternion)

        observation = np.concatenate((hand_observation, cube_observation))
        return observation

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
