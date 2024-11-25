import numpy as np
from time import sleep
import pybullet as p
import pybullet_data

# Limits on movement
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
low = np.concatenate((wrist_low, index_low, middle_low, ring_low, little_low, thumb_low))
high = np.concatenate((wrist_high, index_high, middle_high, ring_high, little_high, thumb_high))

# Connect to the physics client
client = p.connect(p.GUI)
# client = p.connect(p.DIRECT) # Use this to run faster simulations

# Define the gravity for the simulation
p.setGravity(0, 0, -10) 

# Load the base plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Load the simulated hand
startPosition = [0, 0, 1/4]
startOrientation = p.getQuaternionFromEuler([np.pi/2, np.pi, 0])
hand = p.loadURDF("Shadow-Gym/shadow_gym/resources/shadow_hand.urdf", startPosition, startOrientation)
cube = p.loadURDF("Shadow-Gym/shadow_gym/resources/cube.urdf", [0, -1/3.5, 1/3])
cube_texture = p.loadTexture("Shadow-Gym/shadow_gym/resources/cube_texture.jpg")
debug_cube = p.loadURDF("Shadow-Gym/shadow_gym/resources/debug_cube.urdf", [1/3, 1/3, 1/3])
p.changeVisualShape(cube, -1, textureUniqueId=cube_texture)
p.changeVisualShape(debug_cube, -1, rgbaColor=[1.0,0.0,0.0,0.8])
p.changeDynamics(bodyUniqueId=debug_cube, linkIndex=-1, mass=0)
def get_join_info():
    """
    Retrieves each joint in the urdf file and provides some info
    """
    number_of_joints = p.getNumJoints(hand)
    for joint_number in range(number_of_joints):
        info = p.getJointInfo(hand, joint_number)
        print(f"Join ID: {joint_number}")
        print(info)
        print("-------------")


def manipulate_joint(joint_id):
    """
    Used to manipulate a particular joint

    joint number - description
    --------------------------
    1 - wrist motion (horizontal)
    2 - wrist motion (vertical)
    5 - index finger (horizontal)
    6 - index finger base (vertical)
    7 - index finger middle (vertical)
    8 - index finger tip (vertical)
    10 - middle finger (horizontal)
    11 - middle finger base (vertical)
    12 - middle finger middle (vertical)
    13 - middle finger tip (vertical)
    15 - ring finger (horizontal)
    16 - ring finger base (vertical)
    17 - ring finger middle (vertical)
    18 - ring finger tip (vertical)
    20 - little finger grasp
    21 - little finger (horizontal)
    22 - little finger base (vertical)
    23 - little finger middle (vertical)
    24 - little finger tip (vertical)
    26 - thumb rotation
    27 - thumb finger base (vertical)
    28 - thumb finger middle (vertical)
    29 - thumb finger middle (horizontal)
    30 - thumb finger tip (horizontal)
    """
    param = p.addUserDebugParameter('Input Value', -5, 5, 0)
    while True:
        user_input = p.readUserDebugParameter(param)
        p.setJointMotorControl2(hand, joint_id,
                                p.POSITION_CONTROL,
                                targetPosition=user_input)
        p.stepSimulation()
        position, orientation = p.getBasePositionAndOrientation(cube)
        print(position[2])
        sleep(1/60)

def manipulate_all_joints():
    """
    Used to manipulate a particular joint

    joint number - description
    --------------------------
    1 - wrist motion (horizontal)
    2 - wrist motion (vertical)
    5 - index finger (horizontal)
    6 - index finger base (vertical)
    7 - index finger middle (vertical)
    8 - index finger tip (vertical)
    10 - middle finger (horizontal)
    11 - middle finger base (vertical)
    12 - middle finger middle (vertical)
    13 - middle finger tip (vertical)
    15 - ring finger (horizontal)
    16 - ring finger base (vertical)
    17 - ring finger middle (vertical)
    18 - ring finger tip (vertical)
    20 - little finger grasp
    21 - little finger (horizontal)
    22 - little finger base (vertical)
    23 - little finger middle (vertical)
    24 - little finger tip (vertical)
    26 - thumb rotation
    27 - thumb finger base (vertical)
    28 - thumb finger middle (vertical)
    29 - thumb finger middle (horizontal)
    30 - thumb finger tip (horizontal)
    """
    labels = ['wrist motion (horizontal)',
    'wrist motion (vertical)',
    'index finger (horizontal)',
    'index finger base (vertical)',
    'index finger middle (vertical)',
    'index finger tip (vertical)',
    'middle finger (horizontal)',
    'middle finger base (vertical)',
    'middle finger middle (vertical)',
    'middle finger tip (vertical)',
    'ring finger (horizontal)',
    'ring finger base (vertical)',
    'ring finger middle (vertical)',
    'ring finger tip (vertical)',
    'little finger grasp',
    'little finger (horizontal)',
    'little finger base (vertical)',
    'little finger middle (vertical)',
    'little finger tip (vertical)',
    'thumb rotation',
    'thumb finger base (vertical)',
    'thumb finger middle (vertical)',
    'thumb finger middle (horizontal)',
    'thumb finger tip (horizontal)',]
    joint_ids = [1,2,5,6,7,8,10,11,12,13,15,16,17,18,20,21,22,23,24,26,27,28,29,30]
    print("Number of bodies", p.getNumBodies()) # 3 bodies. Cube, hand and plane
    # When considering the middle finger from top to bottom
    # distal, middle, proximal, knuckle

    # Finding and colouring the fingertips of the robot
    # linkIndexes = []
    # linkNames = []
    # for joint_id in joint_ids:
    #     linkName = p.getJointInfo(hand, joint_id)[12]
    #     print(str(linkName))
    #     if str(linkName).endswith("distal'"):
    #         linkNames.append(linkName)
    #         linkIndexes.append(joint_id)
    # print(linkNames)
    # print(linkIndexes)
    colorIndex = 0
    colours = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],  # White 
        [1.0, 1.0, 0.0, 1.0],  # Yellow
    ]
    linkIndexes = [8, 13, 18, 24, 30]
    for linkIndex in linkIndexes:
        p.changeVisualShape(hand, linkIndex=linkIndex, rgbaColor=colours[colorIndex])
        colorIndex += 1
    
    params = []
    for i in range(len(labels)):
        param = p.addUserDebugParameter(labels[i], float(low[i]),  float(high[i]), 0)
        # param = p.addUserDebugParameter(labels[i], -3,  3, 0)
        params.append(param)
    while True:
        for i in range(len(joint_ids)):
            user_input = p.readUserDebugParameter(params[i])
            p.setJointMotorControl2(hand, joint_ids[i], p.POSITION_CONTROL, targetPosition=user_input)
        p.stepSimulation()
        position, orientation = p.getBasePositionAndOrientation(cube)
        # print(f"pos {position} ori {orientation}")
        sleep(1/60)

def stress_test():
    """
    In the URDF files, the joints have a velocity limit (2 or 4). Is it possible for apply_action in shadow_env.py to 
    override these limits?
    """
    # Velocity limits
    wrist_vel_low = np.array([-2.0, -2.0])
    wrist_vel_high = np.array([2.0, 2.0])
    index_vel_low = np.array([-2.0, -2.0, -2.0, -2.0])
    index_vel_high = np.array([2.0, 2.0, 2.0, 2.0])
    middle_vel_low = np.array([-2.0, -2.0, -2.0, -2.0])
    middle_vel_high = np.array([2.0, 2.0, 2.0, 2.0])
    ring_vel_low = np.array([-2.0, -2.0, -2.0, -2.0])
    ring_vel_high = np.array([2.0, 2.0, 2.0, 2.0])
    little_vel_low = np.array([-2.0, -2.0, -2.0, -2.0, -2.0])
    little_vel_high = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    thumb_vel_low = np.array([-4.0, -4.0, -4.0, -4.0, -4.0])
    thumb_vel_high = np.array([4.0, 4.0, 4.0, 4.0, 4.0])
    hand_vel_low = np.concatenate((wrist_vel_low, index_vel_low, middle_vel_low, ring_vel_low, little_vel_low, thumb_vel_low))
    hand_vel_high = np.concatenate((wrist_vel_high, index_vel_high, middle_vel_high, ring_vel_high, little_vel_high, thumb_vel_high))

    # Rotation limits. Actions cannot exceed these limits.
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
    hand_position_low = np.concatenate((wrist_low, index_low, middle_low, ring_low, little_low, thumb_low))
    hand_position_high = np.concatenate((wrist_high, index_high, middle_high, ring_high, little_high, thumb_high))
    
    # Discretize the action space
    number_of_bins = 11
    wrist_bin_size = (wrist_high - wrist_low) / number_of_bins
    index_bin_size = (index_high - index_low) / number_of_bins
    middle_bin_size = (middle_high - middle_low) / number_of_bins
    ring_bin_size = (ring_high - ring_low) / number_of_bins
    little_bin_size = (little_high - little_low) / number_of_bins
    thumb_bin_size = (thumb_high - thumb_low) / number_of_bins
    bin_sizes = np.concatenate(
        (wrist_bin_size, index_bin_size, middle_bin_size, ring_bin_size, little_bin_size, thumb_bin_size,))
    def apply_action(action):
        joints = [1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30]
        for i in range(len(action)):
            joint_id = joints[i]
            target_position = action[i]
            p.setJointMotorControl2(hand, joint_id,
                                p.POSITION_CONTROL,
                                targetPosition=target_position)
            
    # Test each joint with the max and min to see if it exceeds its limit.
    labels = ['wrist motion (horizontal)',
    'wrist motion (vertical)',
    'index finger (horizontal)',
    'index finger base (vertical)',
    'index finger middle (vertical)',
    'index finger tip (vertical)',
    'middle finger (horizontal)',
    'middle finger base (vertical)',
    'middle finger middle (vertical)',
    'middle finger tip (vertical)',
    'ring finger (horizontal)',
    'ring finger base (vertical)',
    'ring finger middle (vertical)',
    'ring finger tip (vertical)',
    'little finger grasp',
    'little finger (horizontal)',
    'little finger base (vertical)',
    'little finger middle (vertical)',
    'little finger tip (vertical)',
    'thumb rotation',
    'thumb finger base (vertical)',
    'thumb finger middle (vertical)',
    'thumb finger middle (horizontal)',
    'thumb finger tip (horizontal)',]
    joints = [1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30]
    for i in range(len(joints)):
        # joint_state[0] is radian position, joint_state[1] is angular velocity (radians / sec)
        # Mininum action
        action = np.zeros(len(joints))
        action[i] = 0 
        # Discretize the action
        action = hand_position_low + (bin_sizes / 2) + (bin_sizes * action)

        print(f"action {action[i]} full {action}")
        apply_action(action)
        joint_id = joints[i]
        for _ in range(20):
            p.stepSimulation()
            joint_state = p.getJointState(hand, joint_id)
            if not (hand_position_low[i]-0.001 <= joint_state[0] <= hand_position_high[i]+0.001):
                print(f"{joint_id} {labels[i]} pos out of bounds. {hand_position_low[i]:.3f} <= {joint_state[0]:.3f} <= {hand_position_high[i]:.3f} is false")
            if not (hand_vel_low[i]-0.001 <= joint_state[1] <= hand_vel_high[i]+0.001):
                print(f"{joint_id} {labels[i]} vel out of bounds. {hand_vel_low[i]:.3f} <= {joint_state[1]} <= {hand_vel_high[i]+0.001}")
        # The above checks are not seen by the observation function. The below represent the observation function
        if not (hand_position_low[i]-0.001 <= joint_state[0] <= hand_position_high[i]+0.001):
                print(f"OBS {joint_id} {labels[i]} pos out of bounds. {hand_position_low[i]:.3f} <= {joint_state[0]:.3f} <= {hand_position_high[i]:.3f} is false")
        if not (hand_vel_low[i]-0.001 <= joint_state[1] <= hand_vel_high[i]+0.001):
                print(f"OBS {joint_id} {labels[i]} vel out of bounds. {hand_vel_low[i]:.3f} <= {joint_state[1]} <= {hand_vel_high[i]+0.001}")
        for _ in range(20):
            p.stepSimulation()
        sleep(1)

        # Maximum action
        action = np.zeros(len(joints))
        action[i] = 11
        # Convert discrete action to continuous motor input.
        action = hand_position_low + (bin_sizes / 2) + (bin_sizes * action)
        print(f"action {action[i]} full {action}")
        apply_action(action)
        for _ in range(20):
            p.stepSimulation()
            joint_state = p.getJointState(hand, joint_id)
            if not (hand_position_low[i]-0.001 <= joint_state[0] <= hand_position_high[i]+0.001):
                print(f"{joint_id} {labels[i]} pos out of bounds. {hand_position_low[i]:.3f} <= {joint_state[0]:.3f} <= {hand_position_high[i]:.3f} is false")
            if not (hand_vel_low[i]-0.001 <= joint_state[1] <= hand_vel_high[i]+0.001):
                print(f"{joint_id} {labels[i]} vel out of bounds. {hand_vel_low[i]:.3f} <= {joint_state[1]} <= {hand_vel_high[i]+0.001}")
        # The above checks are not seen by the observation function. The below represent the observation function
        if not (hand_position_low[i]-0.001 <= joint_state[0] <= hand_position_high[i]+0.001):
                print(f"OBS {joint_id} {labels[i]} pos out of bounds. {hand_position_low[i]:.3f} <= {joint_state[0]:.3f} <= {hand_position_high[i]:.3f} is false")
        if not (hand_vel_low[i]-0.001 <= joint_state[1] <= hand_vel_high[i]+0.001):
                print(f"OBS {joint_id} {labels[i]} vel out of bounds. {hand_vel_low[i]:.3f} <= {joint_state[1]} <= {hand_vel_high[i]+0.001}")
        for _ in range(20):
            p.stepSimulation()
        sleep(1)
    
def run_sim():
    """
    Runs a basic simulation with the loaded urdf files
    """
    while True:
        p.stepSimulation()
        sleep(1/60)

# manipulate_joint(30)
manipulate_all_joints()
# stress_test()
