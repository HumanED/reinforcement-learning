import numpy as np
from time import sleep
import pybullet as p
import pybullet_data

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
hand = p.loadURDF("urdfs/shadow_hand/shadow_hand.urdf", startPosition, startOrientation)
cube = p.loadURDF("urdfs/cube/cube.urdf", [0, -1/3.5, 1/3])

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


def run_sim():
    """
    Runs a basic simulation with the loaded urdf files
    """
    while True:
        p.stepSimulation()
        sleep(1/60)

manipulate_joint(30)
