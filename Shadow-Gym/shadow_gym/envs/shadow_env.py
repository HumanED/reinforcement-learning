import gym
import numpy as np
import pybullet as p
from shadow_gym.resources.hand import Hand
from shadow_gym.resources.plane import Plane
from shadow_gym.resources.cube import Cube
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Changed by Ethan
GUI = True
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


# def get_axis_angle_difference(orientation1, orientation2):
#     """Both orientations must use Quaternions"""
#     rot1 = Rotation.from_quat(orientation1)
#     rot2 = Rotation.from_quat(orientation2)
#
#     # Calculate angular difference (radians) between two rotations
#     # .inv() is transform of the rotation matrix
#     angular_difference = rot1.inv() * rot2
#
#     #  axis-angle representation of angular difference
#     axis_angle = angular_difference.as_rotvec()
#     return axis_angle

def calculate_angular_difference(orientation1, orientation2):
    """Both orientations must use Quaternions"""
    rot1 = Rotation.from_quat(orientation1)
    rot2 = Rotation.from_quat(orientation2)
    # Calculate angular difference (radians) between two rotations
    # .inv() is transform of the rotation matrix
    angular_difference = rot1.inv() * rot2

    #  axis-angle representation of angular difference
    axis_angle = angular_difference.as_rotvec()

    # Convert angular difference into axis-angle representation
    rotation_magnitude = np.linalg.norm(axis_angle)
    return rotation_magnitude


class ShadowEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        if discretize:
            self.action_space = gym.spaces.MultiDiscrete([11] * 24)
        else:
            self.action_space = gym.spaces.box.Box(
                low=hand_motion_low,
                high=hand_motion_high
            )

        self.observation_space = gym.spaces.box.Box(
            # Remeber cube is 12 digit ndarray containing position (x,y,z), orientation in Euler angles (x,y,z), linear velocity (x,y,z) and angular velocity (wx, wy,  wz)
            low=np.concatenate((hand_motion_low, hand_velocity_low, np.array(
                [-10, -10, -10, -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]))),
            high=np.concatenate((hand_motion_high, hand_velocity_high, np.array(
                [10, 10, 10, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])))
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
        self.target_q = p.getQuaternionFromEuler([0, 0, 0])
        self.STEP_LIMIT = 600  # Given a timestep is 1/30 seconds.

        self.reset()

    def step(self, action):
        self.num_steps += 1
        if discretize:
            # Convert discrete number into median of the bins
            action = hand_motion_low + (bin_sizes / 2) + (bin_sizes * action)

        self.hand.apply_action(action)
        p.stepSimulation()

        hand_observation = self.hand.get_observation()
        cube_observation = self.cube.get_observation()

        cube_orientation_q = p.getQuaternionFromEuler(cube_observation[3:6])
        observation = np.concatenate((hand_observation, cube_observation))

        orientation_diff = p.getDifferenceQuaternion(self.target_q, cube_orientation_q)
        orientation_distance = np.linalg.norm(orientation_diff)

        reward = -orientation_distance

        # Reward calculations
        #rotation_to_target = calculate_angular_difference(self.target_q, cube_orientation_q)
        #reward = self.previous_rotation_to_target - rotation_to_target

        if orientation_distance < 0.26:
            # We are less than 15 degrees to the target
            reward += 10
            self.done = True
        if cube_observation[2] < 0.05:
            reward -= 20
            self.done = True
        if self.num_steps > self.STEP_LIMIT:  # 600
            self.done = True
        
        else:
            self.done = False
        #self.previous_rotation_to_target = rotation_to_target

        return observation, reward, self.done, dict()

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        Plane(self.client)
        self.hand = Hand(self.client)
        self.cube = Cube(self.client)

        self.done = False
        self.num_steps = 0

        hand_observation = self.hand.get_observation()
        cube_observation = self.cube.get_observation()

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
