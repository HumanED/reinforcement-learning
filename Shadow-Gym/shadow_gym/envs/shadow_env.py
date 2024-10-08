import gymnasium
import gymnasium.spaces
import gymnasium.spaces.multi_discrete
import numpy as np
import pybullet as p
from shadow_gym.resources.hand import Hand
from shadow_gym.resources.plane import Plane
from shadow_gym.resources.cube import Cube
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import random


discretize = True
number_of_bins = 11

wrist_low = np.array([-0.489, -0.785])
wrist_high = np.array([0.140, 0.524])
index_low = np.array([-0.349, 0.0, 0.0, 0.0])
index_high = np.array([0.349, 1.571, 1.571, 1.571])
middle_low = np.array([-0.349, 0.0, 0.0, 0.0])
middle_high = np.array([0.349, 1.571, 1.571, 1.571])
ring_low = np.array([-0.349, 0.0, 0.0, 0.0])
ring_high = np.array([0.349, 1.571, 1.571, 1.571])
little_low = np.array([0.0, -0.349, 0.0, 0.0, 0.0])
little_high = np.array([0.785, 0.349, 1.571, 1.571, 1.571])
thumb_low = np.array([-0.960, 0.0, -0.209, -0.436, 0.0])
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

cube_pos_low = np.array([-5,-5,-5])
cube_pos_high = np.array([5,5,5])
cube_orientation_q_low = np.array([-1,-1,-1,-1])
cube_orientation_q_high = np.array([1,1,1,1])
cube_relative_q_low = np.array([-1,-1,-1,-1])
cube_relative_q_high = np.array([1,1,1,1])
cube_linear_vel_low = np.array([-np.inf,-np.inf,-np.inf])
cube_linear_vel_high = np.array([np.inf,np.inf,np.inf])
cube_angular_vel_q_low = np.array([-1,-1,-1,-1])
cube_angular_vel_q_high = np.array([1,1,1,1])


def calculate_angular_difference(orientation1, orientation2):
    """
    Calculates angular difference in radians between two quaternions. Quaternion is list of [x, y, z, w]  
    :param list orientation1:  
    :param list orientation1:  
    See  
    Angular difference between quaternions: https://uk.mathworks.com/help/driving/ref/quaternion.dist.html
    Magnitude of axis-angle notation (also known as rotation vector notation) is angle: https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
    Default numpy norn on vectors is euclidean norm https://en.wikipedia.org/wiki/Norm_(mathematics)#:~:text=In%20particular%2C%20the%20Euclidean%20distance,of%20a%20vector%20with%20itself.
    Tested by Ethan 12/9/24
    """
    rot1 = Rotation.from_quat(orientation1)
    rot2 = Rotation.from_quat(orientation2)
    # .inv() is inverse (aka conjugate) of the quaternion
    angular_difference = rot1 * rot2.inv()
    #  axis-angle representation of angular difference
    axis_angle = angular_difference.as_rotvec()
    # Convert angular difference into axis-angle representation
    rotation_magnitude = np.linalg.norm(axis_angle)
    return rotation_magnitude

def angular_velocity_to_quaternion(omega: list[int], delta_t: int=1) -> np.ndarray:
    """
    Converts angular velocity expressed as radians per second [wx, wy, wz] to quaternion [w, x, y, z]
    View https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Describing_rotations_with_quaternions
    and 
    https://math.stackexchange.com/questions/39553/how-do-i-apply-an-angular-velocity-vector3-to-a-unit-quaternion-orientation
    Below is slightly modified version of formula for converting Euler angles to quaternion.
    """
    wx, wy, wz = omega
    omega_mag = np.sqrt(wx**2 + wy**2 + wz**2)
    if omega_mag == 0:
        # No rotation
        return np.array([0,0,0,1])
    theta = omega_mag * delta_t
    ux, uy, uz = wx / omega_mag, wy / omega_mag, wz / omega_mag
    q_w = np.cos(theta / 2)
    q_x = ux * np.sin(theta / 2)
    q_y = uy * np.sin(theta / 2)
    q_z = uz * np.sin(theta / 2)
    return np.array([q_x, q_y, q_z, q_w])


class ShadowEnv(gymnasium.Env):
    """
    :param bool GUI: `GUI=True` after training. `GUI=False` for during training
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, GUI=False):
        if discretize:
            # 11 bins for each of the 24 actions
            self.action_space = gymnasium.spaces.MultiDiscrete(nvec=[11] * 24)
        else:
            self.action_space = gymnasium.spaces.box.Box(
                low=hand_motion_low,
                high=hand_motion_high
            )
        self.observation_space = gymnasium.spaces.Box(
            low=np.concatenate((hand_motion_low, hand_velocity_low, 
                                cube_pos_low, cube_orientation_q_low, cube_relative_q_low, cube_linear_vel_low, cube_angular_vel_q_low)),
            high=np.concatenate((hand_motion_high, hand_velocity_high,
                                cube_pos_high, cube_orientation_q_high, cube_relative_q_high, cube_linear_vel_high, cube_angular_vel_q_high)),
        )

        self.np_random, _ = gymnasium.utils.seeding.np_random()
        if GUI:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setTimeStep(1 / 240, self.client)

        self.hand = None
        self.cube = None
        self.rendered_img = None
        self.num_steps = 0

        self.previous_rotation_to_target = None
        self.target_euler = [0, 0, 0] # Yellow is up
        self.target_quaternion = p.getQuaternionFromEuler(self.target_euler)
        self.STEP_LIMIT = 150
        self.terminated = False
        self.truncated = False
        self.reward = None
        self.info = {}
        self.reset()

    def get_cube_observation(self, target_orientation_q: list[int]) -> np.ndarray:
        """
        Returns 18 digit ndarray containing position (x,y,z), 
        cube orientation quaternion (w, x, y, z),
        relative orientation quaternion to target (w, x, y, z),
        cube velocity (x, y, z) and
        cube angular velocity quaternion (w, x, y, z)
        """
        position, orientation_quaternion = p.getBasePositionAndOrientation(self.cube.cube_body)
        linear_vel, angular_vel = p.getBaseVelocity(self.cube.cube_body)
        angular_vel_q = angular_velocity_to_quaternion(omega=angular_vel, delta_t=1)
        q1 = Rotation.from_quat(target_orientation_q)
        q2 = Rotation.from_quat(orientation_quaternion)
        relative_q = (q1 * q2.inv()).as_quat()
        return np.concatenate((position, orientation_quaternion, relative_q, linear_vel, angular_vel_q),dtype=np.float32)

    def get_hand_observation(self) -> np.ndarray:
        """
        Returns
        24 joint positions (not cartesian x,y,z but a singular angle value in radians. View high and low values and shadow hand docs for more info)
        link velocities relative to Cartesian world (not local frame). View Pybullet docs on getJointState method
        """
        joints = [1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30] #24 joints
        joint_position = []

        # Adding the velocities (x,y,z) of each link
        num_links = p.getNumJoints(self.hand.hand_body)
        link_velocity = []
        for i in range(num_links):
            link_velocity.extend(p.getLinkState(self.hand.hand_body, i, computeLinkVelocity=True)[7])

        for joint_id in joints:
            joint_position.append(p.getJointState(self.hand.hand_body, joint_id)[0])

        return np.array(joint_position + link_velocity, dtype=np.float32)

    def step(self, action):
        self.num_steps += 1
        if discretize:
            # Convert discrete action choice from the AI to a continuous action for the motor.
            action = hand_motion_low + (bin_sizes / 2) + (bin_sizes * action)

        self.hand.apply_action(action)
        # Each simulation step is 4 ms but each environment step has 20 simulation step so is 80 ms of simulation time.
        # Consider simulation time the same as real life time when robot is deployed
        for _ in range(20):
            p.stepSimulation()

        hand_observation = self.get_hand_observation()
        cube_observation = self.get_cube_observation(self.target_quaternion)
        observation = np.concatenate((hand_observation, cube_observation))

        # Reward calculations
        cube_orientation_q = p.getBasePositionAndOrientation(self.cube.cube_body)[1]
        rotation_to_target = calculate_angular_difference(self.target_quaternion, cube_orientation_q)
        if self.previous_rotation_to_target == None:
            self.reward = 0
        else:
            self.reward = self.previous_rotation_to_target - rotation_to_target
        # We are less than 0.4 radians (23 degrees to target)
        if rotation_to_target < 0.4:
            self.reward = 5
            self.info["success"] = 1

        # Termination conditions
        if cube_observation[2] < 0.05:
            # Cube is dropped
            self.reward = -20
            self.terminated = True
        if self.num_steps > self.STEP_LIMIT:  # 300
            self.truncated = True
        self.info["ep_steps"] = self.num_steps
        self.previous_rotation_to_target = rotation_to_target

        return observation, self.reward, self.terminated, self.truncated, self.info

    def reset(self, seed=None, options={}):
        self.seed(seed)
        random_orientation_euler = (random.randint(-1,1) * np.pi/2,
                                    random.randint(-1,1) * np.pi/2,
                                    random.randint(-1,1) * np.pi/2)
        random_orientation_q = p.getQuaternionFromEuler(random_orientation_euler)
        valid_start = False
        while not valid_start:
            valid_start = True
            p.resetSimulation(self.client)
            p.setGravity(0, 0, -10)
            Plane(self.client)
            self.hand = Hand(self.client)
            self.cube = Cube(self.client, random_orientation_q)

        # Apply 30 random actions to further randomise cube start position. If the actions drop the cube, reset the simulation and try again.
            # for _ in range(30):
            #     action = self.action_space.sample()
            #     if discretize:
            #     # Convert discrete action choice from the AI to a continuous action for the motor.
            #         action = hand_motion_low + (bin_sizes / 2) + (bin_sizes * action)
            #     action //= 10
            #     self.hand.apply_action(action)
            #     p.stepSimulation()

            # # Wait for cube to fall a bit
            # for _ in range(5):
            #     p.stepSimulation()
            # # If cube fell, reset simulation
            # cube_observation = self.get_cube_observation(self.target_quaternion)
            # if cube_observation[2] < 0.05:
                # valid_start = False

        # Reset episode termination condition and setup statistics of this episode
        self.terminated = False
        self.truncated = False
        self.num_steps = 0
        self.info = {}
        self.info["ep_steps"] = self.num_steps
        self.info["success"] = 0

        # Initial observation
        hand_observation = self.get_hand_observation()
        cube_observation = self.get_cube_observation(self.target_quaternion)
        observation = np.concatenate((hand_observation, cube_observation))
        return observation, self.info

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
        self.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        return [seed]
